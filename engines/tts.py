"""语音合成。两种引擎，接口相同（文本进，按块产出采样率和 int16 音频）：

- VoiceCloner：GPT-SoVITS，按参考音频克隆音色，支持边合成边输出；
- PresetTts：Kokoro，使用内置的预置音色，不做克隆，模型小、速度快。

依赖：GPT-SoVITS 仓库及其环境（FFmpeg 动态库需在 PATH 中，torchcodec 读音频用）；kokoro、misaki[zh]。
"""

import multiprocessing
import os
import queue
import sys
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 合成函数：给一句文本，逐块产出 (采样率, int16 单声道音频)
SynthesizeStream = Callable[[str], Iterator[tuple[int, np.ndarray]]]


@dataclass
class Voice:
    """一个音色：参考音频加对应文本。克隆音色和预置音色都是这种数据。"""

    ref_audio: str
    ref_text: str
    ref_lang: str = "zh"


def make_clone_engine(sovits_root: str, voice: "Voice", version: str = "v2ProPlus",
                      streaming: bool = True) -> SynthesizeStream:
    """加载声音克隆引擎并预热，返回合成函数。在语音子进程里调用。

    参数:
        streaming: True 时边合成边输出（首块约 0.6 秒），False 时整句合成完才输出
    """
    cloner = VoiceCloner(sovits_root, version)
    cloner.warm_up(voice)
    if streaming:
        return lambda text: cloner.stream(text, voice)
    return lambda text: iter([cloner.synthesize(text, voice)])


def make_preset_engine(model_dir: str, voice_name: str = "zf_001", speed: float = 1.0) -> SynthesizeStream:
    """加载预置音色引擎并预热，返回合成函数。在语音子进程里调用。"""
    preset = PresetTts(model_dir, voice_name, speed)
    preset.synthesize("你好，欢迎来到直播间。")
    return lambda text: iter([preset.synthesize(text)])


def _worker(factory: Callable[[], SynthesizeStream], ffmpeg_dir: str,
            requests: multiprocessing.Queue, responses: multiprocessing.Queue) -> None:
    """语音合成子进程的主循环：加载模型、预热，然后按顺序处理请求。"""
    try:
        os.environ["PATH"] = f"{ffmpeg_dir};{os.environ['PATH']}"  # torchcodec 需要 FFmpeg 动态库
        synthesize = factory()
        responses.put(("ready", None, None, None))
        while (request := requests.get()) is not None:
            key, text = request
            if text is None:  # 标记请求：原样送回，调用方用它判断一批句子是否处理完
                responses.put(("mark", key, None, None))
                continue
            try:
                for sample_rate, chunk in synthesize(text):
                    responses.put(("chunk", key, sample_rate, chunk))
                responses.put(("sentence_end", key, None, None))
            except Exception:
                # 单句失败（例如文本无法转成音素）不应该让整个进程退出
                responses.put(("skipped", key, None, traceback.format_exc()))
    except Exception:
        responses.put(("error", traceback.format_exc(), None, None))


class TtsProcess:
    """在独立进程里运行语音合成。

    语音合成和口型驱动放在同一个进程时会争抢 Python 解释器锁，实测两者都慢一倍左右；
    分成两个进程后互不影响，代价是多占一份 CUDA 上下文的显存。
    请求按提交顺序处理，结果也按同样顺序返回。
    """

    def __init__(self, factory: Callable[[], SynthesizeStream], ffmpeg_dir: str | Path) -> None:
        """
        参数:
            factory: 在子进程里创建合成函数的可调用对象，必须能被 pickle，
                例如 functools.partial(make_clone_engine, ...) 或 functools.partial(make_preset_engine, ...)
            ffmpeg_dir: FFmpeg 动态库所在目录
        """
        self._requests: multiprocessing.Queue = multiprocessing.Queue()
        self._responses: multiprocessing.Queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=_worker, daemon=True, args=(factory, str(ffmpeg_dir), self._requests, self._responses))

    def start(self) -> None:
        """启动子进程并等待模型加载和预热完成。

        抛出:
            RuntimeError: 子进程加载失败
        """
        self._process.start()
        kind, detail, _, _ = self._responses.get()
        if kind != "ready":
            raise RuntimeError(f"语音合成进程启动失败:\n{detail}")

    def submit(self, key, text: str | None) -> None:
        """提交一句要合成的话；text 为 None 时提交一个标记，会按顺序原样返回。"""
        self._requests.put((key, text))

    def result(self, timeout: float = 120.0) -> tuple[str, object, int | None, object]:
        """取下一个结果，阻塞直到有结果。

        返回:
            ("chunk", key, 采样率, 音频块)：一句话的一部分音频，一句话会有一到多块；
            ("sentence_end", key, None, None)：这句话的音频已全部给出；
            ("skipped", key, None, 出错信息)：这句话合成失败，已放弃（之前可能已给出部分音频）；
            ("mark", key, None, None)：submit 提交的标记
        抛出:
            RuntimeError: 子进程加载模型失败、已退出，或超过 timeout 秒没有返回
        """
        waited = 0.0
        while True:
            try:
                kind, key, sample_rate, audio = self._responses.get(timeout=1.0)
                break
            except queue.Empty:
                waited += 1.0
                if not self._process.is_alive():
                    raise RuntimeError("语音合成进程已退出") from None
                if waited >= timeout:
                    raise RuntimeError(f"语音合成超过 {timeout:.0f} 秒没有返回") from None
        if kind == "error":
            raise RuntimeError(f"语音合成出错:\n{key}")
        return kind, key, sample_rate, audio

    def stop(self) -> None:
        """结束子进程。"""
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()


class VoiceCloner:
    """GPT-SoVITS 推理封装。"""

    _WEIGHTS = {
        "v2ProPlus": ("s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
        "v2Pro": ("s1v3.ckpt", "v2Pro/s2Gv2Pro.pth"),
    }

    def __init__(self, sovits_root: str | Path, version: str = "v2ProPlus") -> None:
        """
        参数:
            sovits_root: GPT-SoVITS 仓库根目录
            version: 模型版本。流式输出只支持 v2ProPlus
        抛出:
            KeyError: 不支持的版本

        注意：GPT-SoVITS 内部使用相对路径，这里会把进程工作目录切换到 sovits_root。
        """
        root = Path(sovits_root).resolve()
        os.chdir(root)
        for path in (str(root), str(root / "GPT_SoVITS")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        t2s, vits = self._WEIGHTS[version]
        base = "GPT_SoVITS/pretrained_models"
        self._tts = TTS(TTS_Config({"custom": {
            "device": "cuda",
            "is_half": True,
            "version": version,
            "t2s_weights_path": f"{base}/{t2s}",
            "vits_weights_path": f"{base}/{vits}",
            "bert_base_path": f"{base}/chinese-roberta-wwm-ext-large",
            "cnhuhbert_base_path": f"{base}/chinese-hubert-base",
        }}))

    def warm_up(self, voice: Voice) -> None:
        """用指定音色先合成一句短话，两条合成路径各走一遍。

        首次合成要处理参考音频并初始化各子模型，实测约 16 秒；放在加载阶段做，
        开播后的第一句话就不会卡顿。
        """
        self.synthesize("你好，欢迎来到直播间。", voice)
        for _ in self.stream("你好，欢迎来到直播间。", voice):
            pass

    def _request(self, text: str, voice: Voice, text_lang: str) -> dict:
        if not text.strip():
            raise ValueError("text 不能为空")
        return {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": voice.ref_audio,
            "prompt_text": voice.ref_text,
            "prompt_lang": voice.ref_lang,
            "text_split_method": "cut0",
            "batch_size": 1,
            # 逐句合成时关闭并行推理更快（见 scripts/bench_tts_modes.py）
            "parallel_infer": False,
        }

    def synthesize(self, text: str, voice: Voice, text_lang: str = "zh") -> tuple[int, np.ndarray]:
        """合成一句话，整句完成后返回。

        返回:
            (采样率, int16 单声道音频)
        抛出:
            ValueError: text 为空
        """
        pieces = list(self._tts.run(self._request(text, voice, text_lang)))
        return pieces[0][0], np.concatenate([piece[1] for piece in pieces])

    def stream(self, text: str, voice: Voice, text_lang: str = "zh") -> Iterator[tuple[int, np.ndarray]]:
        """边合成边输出一句话。

        用的是 GPT-SoVITS 的加速语义流式（上游 docs/en/semantic_streaming.md）：
        固定长度的首块让第一段音频尽快出来（实测约 0.6 秒），之后的块逐步加大以减少开销。
        总耗时比整句合成多三到五成，换来更早开口。

        返回:
            逐块产出 (采样率, int16 单声道音频)，各块首尾相接即为整句
        抛出:
            ValueError: text 为空
        """
        request = self._request(text, voice, text_lang) | {
            "streaming_mode": True,
            "gpt_streaming": True,
            "fixed_length_chunk": True,
            "streaming_window_mode": "hybrid",
        }
        for sample_rate, chunk in self._tts.run(request):
            if len(chunk):  # 流式模式会产出空块
                yield sample_rate, chunk


class PresetTts:
    """Kokoro 预置音色语音合成（82M 参数的小模型，内置一百多个中文音色，输出 24 kHz）。"""

    def __init__(self, model_dir: str | Path, voice_name: str = "zf_001", speed: float = 1.0) -> None:
        """
        参数:
            model_dir: Kokoro-82M-v1.1-zh 模型目录（含 config.json、kokoro-v1_1-zh.pth、voices）
            voice_name: 音色名，对应 voices 目录下的文件名，zf 开头为女声，zm 开头为男声
            speed: 语速倍率
        抛出:
            FileNotFoundError: 模型或音色文件不存在
        """
        import torch
        from kokoro import KModel, KPipeline

        model_dir = Path(model_dir)
        voice_path = model_dir / "voices" / f"{voice_name}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(f"找不到音色文件: {voice_path}")
        repo_id = "hexgrad/Kokoro-82M-v1.1-zh"
        model = KModel(repo_id=repo_id, config=str(model_dir / "config.json"),
                       model=str(model_dir / "kokoro-v1_1-zh.pth")).to("cuda").eval()
        self._pipeline = KPipeline(lang_code="z", repo_id=repo_id, model=model)
        self._voice = torch.load(voice_path, weights_only=True)
        self._speed = speed

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """合成一句话。

        返回:
            (采样率, int16 单声道音频)
        抛出:
            ValueError: text 为空
        """
        if not text.strip():
            raise ValueError("text 不能为空")
        pieces = [result.audio.cpu().numpy() for result in self._pipeline(text, voice=self._voice, speed=self._speed)]
        audio = np.concatenate(pieces)
        return 24000, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

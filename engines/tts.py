"""语音合成：封装 GPT-SoVITS，克隆音色和预置音色共用同一接口。

依赖：GPT-SoVITS 仓库及其环境；FFmpeg 动态库需在 PATH 中（torchcodec 读音频用）。
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Voice:
    """一个音色：参考音频加对应文本。克隆音色和预置音色都是这种数据。"""

    ref_audio: str
    ref_text: str
    ref_lang: str = "zh"


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
            version: 模型版本，v2ProPlus 或 v2Pro
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
        """用指定音色先合成一句短话。

        首次合成要处理参考音频并初始化各子模型，实测约 16 秒；放在加载阶段做，
        开播后的第一句话就不会卡顿。
        """
        self.synthesize("你好，欢迎来到直播间。", voice)

    def synthesize(self, text: str, voice: Voice, text_lang: str = "zh") -> tuple[int, np.ndarray]:
        """合成一句话。

        参数:
            text: 要说的内容
            voice: 音色
            text_lang: 内容的语言
        返回:
            (采样率, int16 单声道音频)
        抛出:
            ValueError: text 为空
        """
        if not text.strip():
            raise ValueError("text 不能为空")
        pieces = list(self._tts.run({
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": voice.ref_audio,
            "prompt_text": voice.ref_text,
            "prompt_lang": voice.ref_lang,
            "text_split_method": "cut0",
            "batch_size": 1,
            "parallel_infer": True,
        }))
        return pieces[0][0], np.concatenate([piece[1] for piece in pieces])

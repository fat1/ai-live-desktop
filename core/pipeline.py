"""端到端离线流程：弹幕文件 -> 大模型回复 -> 语音合成 -> 口型驱动 -> 带声音的 mp4。

用来验证整条链路、统计各环节耗时和显存。三个环节并行运行：
生产线程（大模型加语音合成）、主线程（口型的显卡推理）、写出线程（贴回原图加视频编码）。
实时播放和界面在后续里程碑实现。

用法（在项目根目录下，用服务器上的 sovits 环境运行）::

    # 完整流程，并把回复和语音保存成脚本目录
    python -m core.pipeline --danmaku examples/danmaku.txt --out out/demo.mp4 --save_script out/script

    # 直接使用已保存的脚本目录：不加载大模型和语音合成，只做口型驱动
    python -m core.pipeline --from_script out/script --out out/demo2.mp4

依赖：engines 下的三个引擎及其模型；ffmpeg 可执行文件。
"""

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SYSTEM_PROMPT = (
    "你是一名带货主播，正在直播间里和观众聊天。"
    "请用口语化的中文简短回答观众的弹幕，语气亲切自然，每次回答两到三句话，"
    "不要使用表情符号、括号说明或任何格式标记。"
)
FPS = 25


@dataclass
class Segment:
    """一句要播出的话。"""

    text: str
    sample_rate: int
    audio: np.ndarray  # int16 单声道


def split_sentences(text: str, min_chars: int = 6) -> list[str]:
    """把回复按句末标点切成句子，过短的片段并入前一句。

    参数:
        text: 回复全文
        min_chars: 短于该长度的片段不单独成句
    返回:
        句子列表，保留句末标点
    """
    parts = [part.strip() for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text) if part.strip()]
    sentences: list[str] = []
    for part in parts:
        if sentences and len(part) < min_chars:
            sentences[-1] += part
        else:
            sentences.append(part)
    return sentences


def fit_audio(audio: np.ndarray, sample_rate: int, frame_count: int) -> np.ndarray:
    """把音频裁剪或补零到与画面帧数严格等长，避免多句累积后音画错位。"""
    target = round(frame_count * sample_rate / FPS)
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=audio.dtype)])


def gpu_memory_used() -> int:
    """返回整张显卡当前已用显存（MiB），包含所有进程。"""
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout
    return int(output.strip().splitlines()[0])


def generate_segments(danmaku: list[str], llm, cloner, voice, timings: dict,
                      script_dir: Path | None) -> Iterator[Segment]:
    """用大模型和语音合成逐句生成内容；给了 script_dir 时同时把结果存盘。"""
    script = []
    count = 0
    for number, message in enumerate(danmaku, 1):
        start = time.perf_counter()
        reply = llm.chat(SYSTEM_PROMPT, message)
        timings["llm"] += time.perf_counter() - start
        print(f"[{number}/{len(danmaku)}] 弹幕: {message}\n    回复: {reply}", flush=True)
        entry = {"danmaku": message, "reply": reply, "sentences": []}
        for sentence in split_sentences(reply):
            start = time.perf_counter()
            sample_rate, audio = cloner.synthesize(sentence, voice)
            timings["tts"] += time.perf_counter() - start
            if script_dir:
                count += 1
                name = f"{count:04d}.wav"
                sf.write(script_dir / name, audio, sample_rate)
                entry["sentences"].append({"text": sentence, "wav": name})
            yield Segment(sentence, sample_rate, audio)
        script.append(entry)
    if script_dir:
        (script_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")


def load_segments(script_dir: Path) -> Iterator[Segment]:
    """读取已保存的脚本目录，不需要大模型和语音合成。

    抛出:
        FileNotFoundError: 目录里没有 script.json 或缺少音频
    """
    script = json.loads((script_dir / "script.json").read_text(encoding="utf-8"))
    for entry in script:
        for sentence in entry["sentences"]:
            audio, sample_rate = sf.read(script_dir / sentence["wav"], dtype="int16")
            yield Segment(sentence["text"], sample_rate, audio)


def run_in_thread(target, errors: list) -> threading.Thread:
    """启动后台线程；线程里的异常记入 errors，由主线程统一抛出。"""
    def wrapper() -> None:
        try:
            target()
        except BaseException as error:  # 包括 KeyboardInterrupt，统一交给主线程处理
            errors.append(error)

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    return thread


def main() -> None:
    root = Path(os.environ.get("AI_LIVE_ROOT", r"C:\ai_live"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--danmaku", help="弹幕文件，每行一条")
    parser.add_argument("--from_script", help="已保存的脚本目录；给了它就不运行大模型和语音合成")
    parser.add_argument("--save_script", help="把生成的回复和语音保存到该目录")
    parser.add_argument("--out", required=True, help="输出 mp4 路径")
    parser.add_argument("--avatar", default=str(root / "repos/MuseTalk/results/v15/avatars/avator_1"))
    parser.add_argument("--ref_audio", default=str(root / "bench_tts/ref.wav"))
    parser.add_argument("--ref_text", default="Before I chased the way the wind would go, now I hunt a storm of darkness.")
    parser.add_argument("--ref_lang", default="en")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_compile", action="store_true", help="不使用 torch.compile")
    parser.add_argument("--pause", type=float, default=0.4, help="句间停顿秒数")
    parser.add_argument("--llm_context", type=int, default=2048)
    args = parser.parse_args()
    if bool(args.danmaku) == bool(args.from_script):
        parser.error("--danmaku 和 --from_script 必须且只能给一个")

    ffmpeg_dir = root / "tools/ffmpeg_shared/bin"
    os.environ["PATH"] = f"{ffmpeg_dir};{os.environ['PATH']}"  # torchcodec 需要 FFmpeg 动态库
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    avatar_dir = Path(args.avatar).resolve()
    ref_audio = str(Path(args.ref_audio).resolve())
    script_dir = Path(args.save_script).resolve() if args.save_script else None
    if script_dir:
        script_dir.mkdir(parents=True, exist_ok=True)

    # 引擎在解析完所有输入路径之后再加载：语音引擎会切换进程工作目录
    from engines.lipsync import LipSyncEngine

    timings = {"llm": 0.0, "tts": 0.0, "blend_and_encode": 0.0}
    print(f"gpu memory before loading: {gpu_memory_used()} MiB", flush=True)
    load_start = time.perf_counter()
    lipsync = LipSyncEngine(root / "repos/MuseTalk", batch_size=args.batch_size, use_compile=not args.no_compile)
    avatar = lipsync.load_avatar(avatar_dir)
    llm = None
    if args.from_script:
        segments = load_segments(Path(args.from_script).resolve())
    else:
        from engines.llm import LlmServer
        from engines.tts import Voice, VoiceCloner

        danmaku = [line.strip() for line in Path(args.danmaku).resolve().read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        cloner = VoiceCloner(root / "repos/GPT-SoVITS")
        voice = Voice(ref_audio=ref_audio, ref_text=args.ref_text, ref_lang=args.ref_lang)
        cloner.warm_up(voice)
        llm = LlmServer(root / "tools/llama_cuda", root / "models/llm/Qwen3-4B-Q4_K_M.gguf",
                        context_size=args.llm_context)
        llm.start()
        segments = generate_segments(danmaku, llm, cloner, voice, timings, script_dir)
    peak_memory = gpu_memory_used()
    print(f"models loaded in {time.perf_counter() - load_start:.1f}s, gpu memory {peak_memory} MiB", flush=True)

    height, width = avatar.frames[0].shape[:2]
    silent_video = out_path.with_suffix(".video.mp4")
    wav_path = out_path.with_suffix(".wav")
    sentence_wav = out_path.with_suffix(".sentence.wav")
    encoder = subprocess.Popen(
        [str(ffmpeg_dir / "ffmpeg.exe"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", str(silent_video)],
        stdin=subprocess.PIPE)

    errors: list[BaseException] = []
    segment_queue: queue.Queue = queue.Queue(maxsize=8)   # 生产线程 -> 主线程
    frame_queue: queue.Queue = queue.Queue(maxsize=64)    # 主线程 -> 写出线程

    def produce() -> None:
        try:
            for segment in segments:
                segment_queue.put(segment)
        finally:
            segment_queue.put(None)

    def write_frames() -> None:
        while (item := frame_queue.get()) is not None:
            start = time.perf_counter()
            index, face = item
            frame = avatar.frames[index % len(avatar)] if face is None else lipsync.blend(avatar, index, face)
            encoder.stdin.write(frame.tobytes())
            timings["blend_and_encode"] += time.perf_counter() - start

    wall_start = time.perf_counter()
    producer = run_in_thread(produce, errors)
    writer = run_in_thread(write_frames, errors)
    audio_parts: list[np.ndarray] = []
    sample_rate = 0
    cursor = 0  # 素材帧游标，保证各段画面连续
    pause_frames = round(args.pause * FPS)
    lipsync_wait = 0.0  # 主线程等待上游语音的时间
    try:
        while True:
            start = time.perf_counter()
            segment = segment_queue.get()
            lipsync_wait += time.perf_counter() - start
            if segment is None or errors:
                break
            sample_rate = segment.sample_rate
            sf.write(sentence_wav, segment.audio, sample_rate)
            frame_count = 0
            for index, face in lipsync.render_faces(avatar, sentence_wav, start_index=cursor):
                frame_queue.put((index, face))
                frame_count += 1
            audio_parts.append(fit_audio(segment.audio, sample_rate, frame_count))
            cursor += frame_count
            for k in range(pause_frames):  # 句间停顿：播放原始画面
                frame_queue.put((cursor + k, None))
            audio_parts.append(np.zeros(round(pause_frames * sample_rate / FPS), dtype=segment.audio.dtype))
            cursor += pause_frames
            peak_memory = max(peak_memory, gpu_memory_used())
    finally:
        frame_queue.put(None)
        writer.join()
        encoder.stdin.close()
        encoder.wait()
        if llm:
            llm.stop()
        sentence_wav.unlink(missing_ok=True)
    wall = time.perf_counter() - wall_start
    producer.join(timeout=5)
    if errors:
        raise errors[0]

    full_audio = np.concatenate(audio_parts)
    sf.write(wav_path, full_audio, sample_rate)
    subprocess.run([str(ffmpeg_dir / "ffmpeg.exe"), "-y", "-v", "error", "-i", str(silent_video), "-i", str(wav_path),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)], check=True)
    silent_video.unlink()
    wav_path.unlink()

    duration = len(full_audio) / sample_rate
    print(f"output: {out_path} ({duration:.1f}s, {cursor} frames)")
    print(f"wall time: {wall:.1f}s, wall time / output duration = {wall / duration:.2f} "
          "(below 1.0 means faster than real time)")
    print("busy time per stage: " + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in
                                              {**timings, **lipsync.stats}.items()))
    print(f"lipsync thread waited {lipsync_wait:.1f}s for upstream audio")
    print(f"gpu memory peak: {peak_memory} MiB")


if __name__ == "__main__":
    main()

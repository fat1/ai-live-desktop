"""端到端离线流程：弹幕文件 -> 大模型回复 -> 语音合成 -> 口型驱动 -> 带声音的 mp4。

这是第一版"跑通"的命令行流程，用来验证整条链路、统计各环节耗时和显存。
实时播放和界面在后续里程碑实现。

用法（在项目根目录下，用服务器上的 sovits 环境运行）::

    python -m core.pipeline --danmaku examples/danmaku.txt --out C:/ai_live/out/demo.mp4

依赖：engines 下的三个引擎及其模型；ffmpeg 可执行文件。
"""

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SYSTEM_PROMPT = (
    "你是一名带货主播，正在直播间里和观众聊天。"
    "请用口语化的中文简短回答观众的弹幕，语气亲切自然，每次回答两到三句话，"
    "不要使用表情符号、括号说明或任何格式标记。"
)


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


def fit_audio(audio: np.ndarray, sample_rate: int, frame_count: int, fps: int) -> np.ndarray:
    """把音频裁剪或补零到与画面帧数严格等长，避免多句累积后音画错位。"""
    target = round(frame_count * sample_rate / fps)
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=audio.dtype)])


def gpu_memory_used() -> int:
    """返回整张显卡当前已用显存（MiB），包含所有进程。"""
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout
    return int(output.strip().splitlines()[0])


def main() -> None:
    root = Path(os.environ.get("AI_LIVE_ROOT", r"C:\ai_live"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--danmaku", required=True, help="弹幕文件，每行一条")
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

    ffmpeg_dir = root / "tools/ffmpeg_shared/bin"
    os.environ["PATH"] = f"{ffmpeg_dir};{os.environ['PATH']}"  # torchcodec 需要 FFmpeg 动态库
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    danmaku = [line.strip() for line in Path(args.danmaku).read_text(encoding="utf-8").splitlines() if line.strip()]
    avatar_dir = Path(args.avatar).resolve()
    ref_audio = str(Path(args.ref_audio).resolve())

    # 引擎在读完所有输入路径之后再加载：语音引擎会切换进程工作目录
    from engines.lipsync import LipSyncEngine
    from engines.llm import LlmServer
    from engines.tts import Voice, VoiceCloner

    fps = 25
    timings = {"llm": 0.0, "tts": 0.0, "lipsync": 0.0}
    peak_memory = gpu_memory_used()
    print(f"gpu memory before loading: {peak_memory} MiB", flush=True)

    load_start = time.perf_counter()
    lipsync = LipSyncEngine(root / "repos/MuseTalk", batch_size=args.batch_size, use_compile=not args.no_compile)
    avatar = lipsync.load_avatar(avatar_dir)
    cloner = VoiceCloner(root / "repos/GPT-SoVITS")
    voice = Voice(ref_audio=ref_audio, ref_text=args.ref_text, ref_lang=args.ref_lang)
    llm = LlmServer(root / "tools/llama_cuda", root / "models/llm/Qwen3-4B-Q4_K_M.gguf",
                    context_size=args.llm_context)
    llm.start()
    print(f"models loaded in {time.perf_counter() - load_start:.1f}s, gpu memory {gpu_memory_used()} MiB", flush=True)

    height, width = avatar.frames[0].shape[:2]
    silent_video = out_path.with_suffix(".video.mp4")
    wav_path = out_path.with_suffix(".wav")
    sentence_wav = out_path.with_suffix(".sentence.wav")
    encoder = subprocess.Popen(
        [str(ffmpeg_dir / "ffmpeg.exe"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", str(silent_video)],
        stdin=subprocess.PIPE)

    audio_parts: list[np.ndarray] = []
    sample_rate = 0
    cursor = 0  # 素材帧游标，保证各段画面连续
    pause_frames = round(args.pause * fps)
    try:
        for number, message in enumerate(danmaku, 1):
            start = time.perf_counter()
            reply = llm.chat(SYSTEM_PROMPT, message)
            timings["llm"] += time.perf_counter() - start
            print(f"[{number}/{len(danmaku)}] 弹幕: {message}\n    回复: {reply}", flush=True)

            for sentence in split_sentences(reply):
                start = time.perf_counter()
                sample_rate, audio = cloner.synthesize(sentence, voice)
                timings["tts"] += time.perf_counter() - start
                sf.write(sentence_wav, audio, sample_rate)

                start = time.perf_counter()
                frame_count = 0
                for frame in lipsync.render(avatar, sentence_wav, start_index=cursor):
                    encoder.stdin.write(frame.tobytes())
                    frame_count += 1
                timings["lipsync"] += time.perf_counter() - start
                audio_parts.append(fit_audio(audio, sample_rate, frame_count, fps))
                cursor += frame_count

                for frame in lipsync.idle(avatar, cursor, pause_frames):
                    encoder.stdin.write(frame.tobytes())
                audio_parts.append(np.zeros(round(pause_frames * sample_rate / fps), dtype=audio.dtype))
                cursor += pause_frames
                peak_memory = max(peak_memory, gpu_memory_used())
    finally:
        encoder.stdin.close()
        encoder.wait()
        llm.stop()
        sentence_wav.unlink(missing_ok=True)

    full_audio = np.concatenate(audio_parts)
    sf.write(wav_path, full_audio, sample_rate)
    subprocess.run([str(ffmpeg_dir / "ffmpeg.exe"), "-y", "-v", "error", "-i", str(silent_video), "-i", str(wav_path),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)], check=True)
    silent_video.unlink()
    wav_path.unlink()

    duration = len(full_audio) / sample_rate
    work = sum(timings.values())
    print(f"output: {out_path} ({duration:.1f}s, {cursor} frames)")
    print(f"time: llm {timings['llm']:.1f}s, tts {timings['tts']:.1f}s, lipsync {timings['lipsync']:.1f}s, "
          f"total {work:.1f}s")
    print(f"generation time / output duration = {work / duration:.2f} (below 1.0 means faster than real time)")
    print("lipsync breakdown: " + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in lipsync.stats.items()))
    print(f"gpu memory peak: {peak_memory} MiB")


if __name__ == "__main__":
    main()

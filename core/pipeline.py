"""端到端流式流程：弹幕 -> 大模型流式回复 -> 语音合成 -> 口型驱动 -> 带声音的 mp4。

按"弹幕实时到达"的方式运行并计量：每条弹幕在系统空闲时到达，大模型凑够一句就
立刻送去合成语音和口型。输出 mp4 用于检查效果，同时报告两个实时性指标：

- 首帧延迟：弹幕到达到第一帧说话画面生成好的时间；
- 不卡顿延迟：要让整段回复连续播完不中断，最少得等多久再开始播。

进程与线程：语音合成在独立进程；本进程内主线程做口型的显卡推理，
另有回复线程（大模型流式输出并切句）和写出线程（贴回原图加视频编码）。
弹幕目前从文件读取，实时采集后续接入。

用法（在项目根目录下，用服务器上的 sovits 环境运行）::

    python -m core.pipeline --danmaku examples/danmaku.txt --out C:/ai_live/out/demo.mp4

依赖：engines 下的三个引擎及其模型；ffmpeg 可执行文件。
"""

import argparse
import functools
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

SYSTEM_PROMPT = (
    "你是一名带货主播，正在直播间里和观众聊天。"
    "请用口语化的中文简短回答观众的弹幕，语气亲切自然，每次回答两到三句话，"
    "不要使用表情符号、括号说明或任何格式标记。"
)
FPS = 25
SENTENCE_END = re.compile(r"[。！？!?；;\n]")
CLAUSE_END = re.compile(r"[。！？!?；;\n，,、：:]")


def stream_sentences(deltas: Iterable[str], min_chars: int = 6, first_clause: bool = True) -> Iterator[str]:
    """把流式输出的文本片段实时切成句子。

    参数:
        deltas: 大模型逐步产出的文本片段
        min_chars: 短于该长度的句子先不产出，和后面的内容合并
        first_clause: 第一段遇到逗号就切出来，让数字人更早开口；之后仍按整句切
    返回:
        逐句产出，保留句末标点
    """
    buffer = ""
    pattern = CLAUSE_END if first_clause else SENTENCE_END
    for delta in deltas:
        buffer += delta
        search_from = 0
        while (match := pattern.search(buffer, search_from)) is not None:
            if match.end() >= min_chars:
                sentence = buffer[:match.end()].strip()
                buffer = buffer[match.end():]
                search_from = 0
                if sentence:
                    pattern = SENTENCE_END  # 只有第一段按逗号切，之后恢复按句切
                    yield sentence
            else:
                search_from = match.end()
    if buffer.strip():
        yield buffer.strip()


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
    parser.add_argument("--gap", type=float, default=1.0, help="两条回复之间的待机秒数")
    parser.add_argument("--llm_context", type=int, default=2048)
    parser.add_argument("--preset_voice", help="使用预置音色（不做声音克隆），给出音色名，例如 zf_001")
    parser.add_argument("--tts_parallel", action="store_true", help="开启 GPT-SoVITS 的并行推理")
    parser.add_argument("--no_first_clause", action="store_true", help="第一句不按逗号提前切分")
    parser.add_argument("--deterministic", action="store_true", help="大模型温度设为 0，便于多次运行对比")
    args = parser.parse_args()

    from engines.lipsync import LipSyncEngine
    from engines.llm import LlmServer
    from engines.tts import TtsProcess, Voice, make_clone_engine, make_preset_engine

    ffmpeg_dir = root / "tools/ffmpeg_shared/bin"
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    danmaku = [line.strip() for line in Path(args.danmaku).read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"gpu memory before loading: {gpu_memory_used()} MiB", flush=True)
    load_start = time.perf_counter()
    voice = Voice(ref_audio=str(Path(args.ref_audio).resolve()), ref_text=args.ref_text, ref_lang=args.ref_lang)
    if args.preset_voice:
        factory = functools.partial(make_preset_engine, str(root / "models/kokoro-zh"), args.preset_voice)
    else:
        factory = functools.partial(make_clone_engine, str(root / "repos/GPT-SoVITS"), voice,
                                    parallel_infer=args.tts_parallel)
    tts = TtsProcess(factory, ffmpeg_dir)
    temperature = 0.0 if args.deterministic else 0.7
    llm = LlmServer(root / "tools/llama_cuda", root / "models/llm/Qwen3-4B-Q4_K_M.gguf",
                    context_size=args.llm_context)
    # 三个模型的加载互不依赖：语音在子进程里加载，大模型是外部进程，本进程同时加载口型
    loaders = [threading.Thread(target=tts.start), threading.Thread(target=llm.start)]
    for loader in loaders:
        loader.start()
    lipsync = LipSyncEngine(root / "repos/MuseTalk", batch_size=args.batch_size, use_compile=not args.no_compile)
    avatar = lipsync.load_avatar(Path(args.avatar).resolve())
    for loader in loaders:
        loader.join()
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
    frame_queue: queue.Queue = queue.Queue(maxsize=64)  # 主线程 -> 写出线程
    ready_times: list[float] = []  # 当前回复每一帧生成好的时刻

    def write_frames() -> None:
        try:
            while (item := frame_queue.get()) is not None:
                if isinstance(item, threading.Event):  # 同步标记：之前的帧都已写出
                    item.set()
                    continue
                index, face = item
                frame = avatar.frames[index % len(avatar)] if face is None else lipsync.blend(avatar, index, face)
                encoder.stdin.write(frame.tobytes())
                ready_times.append(time.perf_counter())
        except BaseException as error:
            errors.append(error)

    def flush_frames() -> None:
        """等待写出线程处理完已提交的所有帧。"""
        done = threading.Event()
        frame_queue.put(done)
        while not done.wait(timeout=1):
            if errors:
                raise errors[0]

    writer = threading.Thread(target=write_frames, daemon=True)
    writer.start()
    audio_parts: list[np.ndarray] = []
    sample_rate = 32000
    cursor = 0  # 素材帧游标，保证各段画面连续
    pause_frames = round(args.pause * FPS)
    metrics = []
    try:
        for number, message in enumerate(danmaku, 1):
            ready_times.clear()
            reply_sentences: list[str] = []
            arrival = time.perf_counter()
            marks = {}

            def reply_worker(number=number, message=message, marks=marks, reply_sentences=reply_sentences) -> None:
                try:
                    for sentence in stream_sentences(llm.chat_stream(SYSTEM_PROMPT, message, temperature=temperature),
                                                     first_clause=not args.no_first_clause):
                        marks.setdefault("first_sentence", time.perf_counter())
                        reply_sentences.append(sentence)
                        tts.submit(number, sentence)
                except BaseException as error:
                    errors.append(error)
                finally:
                    tts.submit(number, None)  # 标记这条回复的句子已全部提交

            threading.Thread(target=reply_worker, daemon=True).start()
            reply_frames = 0
            while True:
                kind, _, rate, audio = tts.result()
                if errors:
                    raise errors[0]
                if kind == "mark":
                    break
                marks.setdefault("first_audio", time.perf_counter())
                sample_rate = rate
                sf.write(sentence_wav, audio, rate)
                frame_count = 0
                for index, face in lipsync.render_faces(avatar, sentence_wav, start_index=cursor):
                    frame_queue.put((index, face))
                    frame_count += 1
                audio_parts.append(fit_audio(audio, rate, frame_count))
                cursor += frame_count
                for k in range(pause_frames):  # 句间停顿：播放原始画面
                    frame_queue.put((cursor + k, None))
                audio_parts.append(np.zeros(round(pause_frames * rate / FPS), dtype=np.int16))
                cursor += pause_frames
                reply_frames += frame_count + pause_frames
            flush_frames()
            peak_memory = max(peak_memory, gpu_memory_used())

            if ready_times:
                offsets = [moment - arrival for moment in ready_times]
                metrics.append({
                    "first_sentence": marks["first_sentence"] - arrival,
                    "first_audio": marks["first_audio"] - arrival,
                    "first_frame": offsets[0],
                    # 第 i 帧在开播后 i/FPS 秒才需要，取所有帧里最晚的"生成时刻减播放时刻"
                    "no_stall": max(offset - i / FPS for i, offset in enumerate(offsets)),
                    "duration": reply_frames / FPS,
                    "generation": offsets[-1],
                })
                m = metrics[-1]
                print(f"[{number}/{len(danmaku)}] 弹幕: {message}\n    回复: {''.join(reply_sentences)}\n"
                      f"    first sentence {m['first_sentence']:.2f}s, first audio {m['first_audio']:.2f}s, "
                      f"first frame {m['first_frame']:.2f}s, no-stall delay {m['no_stall']:.2f}s, "
                      f"generated {m['duration']:.1f}s of video in {m['generation']:.1f}s", flush=True)

            gap_frames = round(args.gap * FPS)  # 回复之间的待机画面，不计入指标
            for k in range(gap_frames):
                frame_queue.put((cursor + k, None))
            audio_parts.append(np.zeros(round(gap_frames * sample_rate / FPS), dtype=np.int16))
            cursor += gap_frames
            flush_frames()
    finally:
        frame_queue.put(None)
        writer.join(timeout=30)
        encoder.stdin.close()
        encoder.wait()
        llm.stop()
        tts.stop()
        sentence_wav.unlink(missing_ok=True)

    full_audio = np.concatenate(audio_parts)
    sf.write(wav_path, full_audio, sample_rate)
    subprocess.run([str(ffmpeg_dir / "ffmpeg.exe"), "-y", "-v", "error", "-i", str(silent_video), "-i", str(wav_path),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)], check=True)
    silent_video.unlink()
    wav_path.unlink()

    def average(name: str) -> float:
        return sum(m[name] for m in metrics) / len(metrics)

    print(f"output: {out_path} ({len(full_audio) / sample_rate:.1f}s, {cursor} frames)")
    print(f"average over {len(metrics)} replies: first sentence {average('first_sentence'):.2f}s, "
          f"first audio {average('first_audio'):.2f}s, first frame {average('first_frame'):.2f}s, "
          f"no-stall delay {average('no_stall'):.2f}s (worst {max(m['no_stall'] for m in metrics):.2f}s)")
    total_generation = sum(m["generation"] for m in metrics)
    total_duration = sum(m["duration"] for m in metrics)
    print(f"generation time / video duration = {total_generation / total_duration:.2f} "
          "(below 1.0 means faster than real time)")
    print(f"gpu memory peak: {peak_memory} MiB")


if __name__ == "__main__":
    main()

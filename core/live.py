"""实时直播流程：弹幕按时间到达，数字人边生成边播。

结构：
- 播放端（core/player.py）：按 25 fps 的时钟出画面和声音，没有说话帧时播待机画面；
- 回复线程：取弹幕，大模型流式生成回复，切句、清洗后交给语音合成；
- 主线程：接收语音合成按块返回的音频，增量生成口型，把"一帧画面加它的声音"放进播放队列；
- 语音合成在独立进程，大模型是外部进程。

这个程序按真实时间运行：弹幕按设定的间隔到达，输出的视频时长等于运行时长，
录下来的就是观众会看到的内容，包括等待时的待机画面和生成跟不上时的停顿。

用法（在项目根目录下，用服务器上的 sovits 环境运行）::

    python -m core.live --danmaku examples/danmaku_hongzao.txt --persona_file examples/persona_laochen.txt
        --avatar <素材目录> --ref_audio ref.wav --ref_text "参考音频的文本" --out C:/ai_live/out/live.mp4

弹幕目前从文件读取并按固定间隔投放，直播间实时采集后续接入。
依赖：engines 下的三个引擎及其模型；ffmpeg 可执行文件。
"""

import argparse
import functools
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from core.player import Player, SpeechFrame
from core.text import DEFAULT_PERSONA, clean_for_speech, stream_sentences

FPS = 25
OUTPUT_RATE = 32000  # 输出声音的采样率；语音引擎的输出不是这个采样率时会重采样
SAMPLES_PER_FRAME = OUTPUT_RATE // FPS


class FileSink:
    """把画面和声音录成 mp4，用于检查效果。"""

    def __init__(self, path: Path, width: int, height: int, ffmpeg: Path) -> None:
        self._path = path
        self._ffmpeg = ffmpeg
        self._video = path.with_suffix(".video.mp4")
        self._audio: list[np.ndarray] = []
        self._encoder = subprocess.Popen(
            [str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", str(self._video)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray, audio: np.ndarray) -> None:
        self._encoder.stdin.write(frame.tobytes())
        self._audio.append(audio)

    def close(self) -> float:
        """结束录制并合成最终文件，返回视频时长（秒）。"""
        self._encoder.stdin.close()
        self._encoder.wait()
        wav = self._path.with_suffix(".wav")
        audio = np.concatenate(self._audio)
        sf.write(wav, audio, OUTPUT_RATE)
        subprocess.run([str(self._ffmpeg), "-y", "-v", "error", "-i", str(self._video), "-i", str(wav),
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(self._path)], check=True)
        self._video.unlink()
        wav.unlink()
        return len(audio) / OUTPUT_RATE


def gpu_memory_used() -> int:
    """返回整张显卡当前已用显存（MiB），包含所有进程。"""
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True).stdout
    return int(output.strip().splitlines()[0])


def main() -> None:
    root = Path(os.environ.get("AI_LIVE_ROOT", r"C:\ai_live"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--danmaku", required=True, help="弹幕文件，每行一条")
    parser.add_argument("--interval", type=float, default=18.0, help="相邻两条弹幕到达的间隔秒数")
    parser.add_argument("--max_wait", type=float, default=15.0, help="弹幕等待超过这么多秒还没轮到就放弃")
    parser.add_argument("--out", required=True, help="录制输出的 mp4 路径")
    parser.add_argument("--avatar", required=True, help="MuseTalk 预处理好的素材目录")
    parser.add_argument("--original_idle", action="store_true",
                        help="待机时直接播素材原始帧。默认待机也用模型生成的闭嘴画面：这样待机和说话时的嘴出自"
                             "同一来源，切换时看不出变化；原始帧更清晰，但一开口嘴的样子会突变")
    parser.add_argument("--persona_file", help="人设文件，内容作为大模型的系统提示词")
    parser.add_argument("--ref_audio", help="声音克隆的参考音频")
    parser.add_argument("--ref_text", help="参考音频的文本")
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--preset_voice", help="使用预置音色（不做声音克隆），给出音色名，例如 zf_001")
    parser.add_argument("--no_tts_streaming", action="store_true", help="语音整句合成完再输出")
    parser.add_argument("--no_enhance", action="store_true", help="不做嘴部超分增强")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pause", type=float, default=0.3, help="句间停顿秒数")
    parser.add_argument("--lead", type=float, default=0.3, help="新一句话的画面比当前播放位置提前多少秒开始排")
    parser.add_argument("--llm_gpu_layers", type=int, default=99, help="大模型放到显卡上的层数，0 表示纯 CPU")
    parser.add_argument("--deterministic", action="store_true", help="大模型温度设为 0，便于多次运行对比")
    args = parser.parse_args()
    if not args.preset_voice and not (args.ref_audio and args.ref_text):
        parser.error("声音克隆需要 --ref_audio 和 --ref_text；或者用 --preset_voice")

    from engines.lipsync import LipSyncEngine
    from engines.llm import LlmServer
    from engines.tts import TtsProcess, Voice, make_clone_engine, make_preset_engine

    ffmpeg_dir = root / "tools/ffmpeg_shared/bin"
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    danmaku = [line.strip() for line in Path(args.danmaku).read_text(encoding="utf-8").splitlines() if line.strip()]
    persona = Path(args.persona_file).read_text(encoding="utf-8").strip() if args.persona_file else DEFAULT_PERSONA
    temperature = 0.0 if args.deterministic else 0.7

    load_start = time.perf_counter()
    if args.preset_voice:
        factory = functools.partial(make_preset_engine, str(root / "models/kokoro-zh"), args.preset_voice)
    else:
        voice = Voice(ref_audio=str(Path(args.ref_audio).resolve()), ref_text=args.ref_text, ref_lang=args.ref_lang)
        factory = functools.partial(make_clone_engine, str(root / "repos/GPT-SoVITS"), voice,
                                    streaming=not args.no_tts_streaming)
    tts = TtsProcess(factory, ffmpeg_dir)
    llm = LlmServer(root / "tools/llama_cuda", root / "models/llm/Qwen3-4B-Q4_K_M.gguf",
                    gpu_layers=args.llm_gpu_layers)
    # 三个模型的加载互不依赖：语音在子进程里加载，大模型是外部进程，本进程同时加载口型
    loaders = [threading.Thread(target=tts.start), threading.Thread(target=llm.start)]
    for loader in loaders:
        loader.start()
    enhance_weights = root / "models/sr/realesr-general-x4v3.pth"
    lipsync = LipSyncEngine(root / "repos/MuseTalk", batch_size=args.batch_size,
                            enhance_weights=None if args.no_enhance else enhance_weights)
    avatar = lipsync.load_avatar(Path(args.avatar).resolve(), closed_mouth=not args.original_idle)
    for loader in loaders:
        loader.join()
    peak_memory = gpu_memory_used()
    print(f"models loaded in {time.perf_counter() - load_start:.1f}s, gpu memory {peak_memory} MiB", flush=True)

    height, width = avatar.frames[0].shape[:2]
    sink = FileSink(out_path, width, height, ffmpeg_dir / "ffmpeg.exe")
    player = Player(avatar, sink, OUTPUT_RATE, FPS, pause_frames=round(args.pause * FPS))
    errors: list[BaseException] = []
    pending: queue.Queue = queue.Queue()     # 已到达、等待回复的弹幕
    reply_finished = threading.Event()       # 上一条回复的画面已全部生成
    reply_finished.set()
    arrivals: dict[int, float] = {}
    picked: dict[int, float] = {}
    texts: dict[int, list[str]] = {}
    dropped: list[int] = []
    session_start = time.perf_counter()

    def deliver_danmaku() -> None:
        """按固定间隔投放弹幕，模拟直播间里弹幕实时到达。"""
        for number, message in enumerate(danmaku, 1):
            time.sleep(max(0.0, session_start + 2.0 + (number - 1) * args.interval - time.perf_counter()))
            arrivals[number] = time.perf_counter()
            pending.put((number, message))
        pending.put(None)

    def reply_worker() -> None:
        """逐条取弹幕，生成回复并提交语音合成。"""
        try:
            while (item := pending.get()) is not None:
                number, message = item
                # 上一条的画面生成完、且积压的画面快播完了，才开始下一条，避免越积越多
                while not (reply_finished.is_set() and player.backlog() <= FPS):
                    if errors:
                        return
                    time.sleep(0.05)
                if time.perf_counter() - arrivals[number] > args.max_wait:
                    dropped.append(number)  # 等太久，观众早就不等了
                    continue
                picked[number] = time.perf_counter()
                reply_finished.clear()
                texts[number] = []
                for sentence in stream_sentences(llm.chat_stream(persona, message, temperature=temperature)):
                    sentence = clean_for_speech(sentence)
                    if sentence:
                        texts[number].append(sentence)
                        tts.submit(number, sentence)
                tts.submit(number, None)  # 标记这条回复的句子已全部提交
        except BaseException as error:
            errors.append(error)
        finally:
            tts.submit(None, None)  # 标记全部弹幕处理完

    threads = [threading.Thread(target=deliver_danmaku, daemon=True), threading.Thread(target=reply_worker, daemon=True)]
    player.start()
    for thread in threads:
        thread.start()

    pause_frames = round(args.pause * FPS)
    lead_frames = round(args.lead * FPS)
    next_start = 0           # 下一句话最早可以从素材的哪一帧开始
    sentence = None          # 正在生成的这句话
    sentence_start = 0
    sentence_audio = np.zeros(0, dtype=np.int16)

    def enqueue(reply: int, faces: list) -> None:
        """把新生成的帧连同各自那 40 毫秒声音放进播放队列。"""
        first = sentence.frames_done - len(faces)
        for offset, (_, face) in enumerate(faces):
            number = first + offset
            audio = sentence_audio[number * SAMPLES_PER_FRAME:(number + 1) * SAMPLES_PER_FRAME]
            if len(audio) < SAMPLES_PER_FRAME:
                audio = np.pad(audio, (0, SAMPLES_PER_FRAME - len(audio)))
            player.put(SpeechFrame(sentence_start + number, face, audio, reply))

    try:
        while True:
            kind, key, rate, payload = tts.result()
            if errors or player.error:
                raise errors[0] if errors else player.error
            if kind == "mark":
                if key is None:
                    break  # 全部弹幕处理完
                player.finish_reply(key)
                reply_finished.set()
                peak_memory = max(peak_memory, gpu_memory_used())
            elif kind == "chunk":
                chunk = payload if rate == OUTPUT_RATE else (librosa.resample(
                    payload.astype(np.float32) / 32768, orig_sr=rate, target_sr=OUTPUT_RATE) * 32767).astype(np.int16)
                if sentence is None:
                    # 新一句话接在上一句之后；如果生成落后了，就从当前播放位置稍后一点接上
                    sentence_start = max(next_start, player.cursor + lead_frames)
                    sentence = lipsync.stream(avatar, OUTPUT_RATE, sentence_start)
                    sentence_audio = np.zeros(0, dtype=np.int16)
                sentence_audio = np.concatenate([sentence_audio, chunk])
                enqueue(key, sentence.feed(chunk))
            elif sentence is not None:  # sentence_end，或这句话中途合成失败
                enqueue(key, sentence.finish())
                next_start = sentence_start + sentence.frames_done + pause_frames
                sentence = None
            if kind == "skipped":
                print(f"skipped one sentence: {str(payload).strip().splitlines()[-1]}", flush=True)

        if errors:
            raise errors[0]
        # 等播放队列播完，再留一秒待机画面收尾
        while player.backlog() and not player.error:
            time.sleep(0.05)
        time.sleep(1.0)
    finally:
        player.stop()
        duration = sink.close()
        llm.stop()
        tts.stop()

    print(f"output: {out_path} ({duration:.1f}s), wall time {time.perf_counter() - session_start:.1f}s")
    latencies = []
    for number, message in enumerate(danmaku, 1):
        if number in dropped:
            print(f"[{number}] 弹幕: {message}\n    dropped after waiting more than {args.max_wait:.0f}s")
            continue
        stats = player.stats[number]
        latency = stats.first_played - arrivals[number]
        latencies.append(latency)
        jumps = f", picture jumped back {len(stats.jumps)} times (max {max(stats.jumps)} frames)" if stats.jumps else ""
        print(f"[{number}] 弹幕: {message}\n    回复: {''.join(texts[number])}\n"
              f"    waited in queue {picked[number] - arrivals[number]:.2f}s, started speaking after {latency:.2f}s, "
              f"spoke {stats.frames / FPS:.1f}s, stalled {stats.stall_frames / FPS:.2f}s{jumps}")
    if latencies:
        total_stall = sum(player.stats[n].stall_frames for n in player.stats) / FPS
        print(f"response latency: average {sum(latencies) / len(latencies):.2f}s, worst {max(latencies):.2f}s; "
              f"total stall while speaking {total_stall:.2f}s; dropped {len(dropped)} of {len(danmaku)}")
    print(f"player: {player.frames_played} frames, {player.late_frames} more than one frame late, "
          f"max lag behind clock {player.max_lag * 1000:.0f} ms")
    print("lipsync busy time: " + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in lipsync.stats.items()))
    print(f"gpu memory peak: {peak_memory} MiB")


if __name__ == "__main__":
    main()

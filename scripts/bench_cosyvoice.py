"""CosyVoice 3 试听与基准：声音克隆、流式输出延迟、情绪指令。

用法（工作目录必须是 CosyVoice 仓库根目录，用 cosyvoice 环境运行）::

    python bench_cosyvoice.py --model C:/ai_live/models/Fun-CosyVoice3-0.5B --ref_audio ref.wav
        --ref_text "参考音频的文本" --sentences sentences.txt --out_dir out

每种模式的合成结果存为 wav 供试听。
依赖：CosyVoice 仓库及其环境。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

SYSTEM = "You are a helpful assistant."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--sentences", required=True, help="每行一句的文本文件")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--instruct", default="请用热情、亲切、有感染力的直播带货语气说这句话。")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences = [line.strip() for line in Path(args.sentences).read_text(encoding="utf-8").splitlines() if line.strip()]

    model = AutoModel(model_dir=args.model)
    print(f"gpu: {torch.cuda.get_device_name(0)}, sample rate {model.sample_rate}")
    print(f"vram after load: {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated")

    prompt = f"{SYSTEM}<|endofprompt|>{args.ref_text}"
    modes = {
        "clone": lambda text, stream: model.inference_zero_shot(text, prompt, args.ref_audio, stream=stream),
        "instruct": lambda text, stream: model.inference_instruct2(
            text, f"{SYSTEM} {args.instruct}<|endofprompt|>", args.ref_audio, stream=stream),
    }

    def run(mode: str, text: str, stream: bool) -> tuple[float, float, torch.Tensor]:
        """返回 (首段出声时间, 总耗时, 音频)。"""
        start = time.perf_counter()
        first = None
        pieces = []
        for piece in modes[mode](text, stream):
            if first is None:
                first = time.perf_counter() - start
            pieces.append(piece["tts_speech"])
        return first, time.perf_counter() - start, torch.cat(pieces, dim=1)

    run("clone", sentences[0], False)  # 预热
    torch.cuda.reset_peak_memory_stats()
    print("mode     | stream | first chunk s | total s | audio s |  RTF")
    for mode in modes:
        for stream in (False, True):
            first_sum = total = audio_seconds = 0.0
            for index, text in enumerate(sentences):
                first, elapsed, audio = run(mode, text, stream)
                first_sum += first
                total += elapsed
                audio_seconds += audio.shape[1] / model.sample_rate
                if not stream:
                    torchaudio.save(str(out_dir / f"cosyvoice3_{mode}_{index}.wav"), audio, model.sample_rate)
            print(f"{mode:8s} | {stream!s:6s} | {first_sum / len(sentences):13.2f} | {total:7.2f} | "
                  f"{audio_seconds:7.2f} | {total / audio_seconds:4.2f}", flush=True)
    print(f"peak alloc: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
          f"peak reserved: {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()

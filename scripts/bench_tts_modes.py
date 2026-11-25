"""对比 GPT-SoVITS 不同设置下的首段出声时间和总耗时，用于降低直播回复的延迟。

用法（工作目录必须是 GPT-SoVITS 仓库根目录）::

    python bench_tts_modes.py --ref_audio ref.wav --ref_text "..." --ref_lang en

依赖：GPT-SoVITS 仓库自带的环境；FFmpeg 动态库在 PATH 中。
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

# 长度接近直播回复里的单句
SENTENCES = [
    "这个面霜是温和配方，敏感肌可以用的。",
    "一般当天发，您下单后大概二十四小时内就能发货啦。",
    "现在下单确实有优惠，价格比平时低不少。",
    "谢谢夸奖啊！",
]
MODES = {
    "default": {},
    "streaming": {"streaming_mode": True},
    "streaming_fixed": {"streaming_mode": True, "fixed_length_chunk": True},
    "fragment": {"return_fragment": True},
    "no_parallel": {"parallel_infer": False},
    # 加速语义流式（docs/en/semantic_streaming.md），对应 API 的 streaming_mode 2 和 3
    "gpt_stream_2": {"streaming_mode": True, "gpt_streaming": True},
    "gpt_stream_3": {"streaming_mode": True, "gpt_streaming": True, "fixed_length_chunk": True},
    "gpt_stream_2_hybrid": {"streaming_mode": True, "gpt_streaming": True, "streaming_window_mode": "hybrid"},
    "gpt_stream_3_hybrid": {"streaming_mode": True, "gpt_streaming": True, "fixed_length_chunk": True,
                            "streaming_window_mode": "hybrid"},
}


def build(version: str, is_half: bool) -> TTS:
    """加载指定精度的模型。"""
    base = "GPT_SoVITS/pretrained_models"
    return TTS(TTS_Config({"custom": {
        "device": "cuda",
        "is_half": is_half,
        "version": version,
        "t2s_weights_path": f"{base}/s1v3.ckpt",
        "vits_weights_path": f"{base}/v2Pro/s2Gv2ProPlus.pth" if version == "v2ProPlus" else f"{base}/v2Pro/s2Gv2Pro.pth",
        "bert_base_path": f"{base}/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": f"{base}/chinese-hubert-base",
    }}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--ref_lang", default="zh")
    parser.add_argument("--versions", default="v2ProPlus,v2Pro")
    parser.add_argument("--half", default="True,False", help="要测的精度，逗号分隔")
    parser.add_argument("--modes", default=",".join(MODES), help="要测的模式，逗号分隔")
    parser.add_argument("--out_dir", help="给了就把每种模式第一句的合成结果存成 wav 供试听")
    args = parser.parse_args()

    print("version   | half  | mode            | first chunk s | total s | audio s |  RTF")
    for version in args.versions.split(","):
        for is_half in [value == "True" for value in args.half.split(",")]:
            tts = build(version, is_half)

            def run(text: str, options: dict, save_as: str | None = None) -> tuple[float, float, float]:
                """返回 (首段出声时间, 总耗时, 音频时长)。"""
                chunks = []
                start = time.perf_counter()
                first = None
                samples = 0
                rate = 1
                for rate, chunk in tts.run({
                    "text": text, "text_lang": "zh", "ref_audio_path": args.ref_audio,
                    "prompt_text": args.ref_text, "prompt_lang": args.ref_lang,
                    "text_split_method": "cut0", "batch_size": 1, "parallel_infer": True, "seed": 42,
                    **options,
                }):
                    if len(chunk) == 0:  # 流式模式会产出空块
                        continue
                    if first is None:
                        first = time.perf_counter() - start
                    samples += len(chunk)
                    chunks.append(chunk)
                if save_as:
                    sf.write(save_as, np.concatenate(chunks), rate)
                return first, time.perf_counter() - start, samples / rate

            run(SENTENCES[0], {})  # 预热
            for mode in args.modes.split(","):
                options = MODES[mode]
                try:
                    results = [run(text, options, os.path.join(args.out_dir, f"{version}_{mode}_{index}.wav")
                                   if args.out_dir and is_half else None)
                               for index, text in enumerate(SENTENCES)]
                except Exception as error:  # 某个模式在该版本上不可用时继续测其余模式
                    print(f"{version:9s} | {is_half!s:5s} | {mode:15s} | failed: {type(error).__name__}: {str(error)[:80]}",
                          flush=True)
                    continue
                first = sum(r[0] for r in results) / len(results)
                total = sum(r[1] for r in results)
                audio = sum(r[2] for r in results)
                print(f"{version:9s} | {is_half!s:5s} | {mode:15s} | {first:13.2f} | {total:7.2f} | {audio:7.2f} | "
                      f"{total / audio:4.2f}", flush=True)
            del tts
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

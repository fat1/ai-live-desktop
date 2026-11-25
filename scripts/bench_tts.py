"""GPT-SoVITS 声音克隆基准测试：合成速度、显存占用，并保存合成结果供试听。

用法（工作目录必须是 GPT-SoVITS 仓库根目录）::

    python bench_tts.py --ref_audio ref.wav --out_dir C:/ai_live/bench_tts

参考音频需为 3 到 10 秒的干净人声。不提供 --ref_text 时用 FunASR 自动识别。
依赖：GPT-SoVITS 仓库自带的环境。
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

SENTENCES = [
    "欢迎新进直播间的朋友们，点点关注不迷路。",
    "这款面霜是我们家卖得最好的，干皮油皮都可以用。",
    "刚才有朋友问发货时间，下单后四十八小时内发出，全国包邮。",
    "现在下单还送同款小样，库存只剩最后两百件了，喜欢的抓紧时间。",
    "感谢大家的支持，有什么问题直接打在公屏上，我看到都会回复。",
]


def transcribe(audio_path: str) -> str:
    """用 FunASR 识别参考音频的文本。

    参数:
        audio_path: 音频文件路径
    返回:
        识别出的文本
    """
    from funasr import AutoModel

    model = AutoModel(model="paraformer-zh", punc_model="ct-punc", disable_update=True, device="cpu")
    return model.generate(input=audio_path)[0]["text"].replace(" ", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", required=True, help="参考音频，3 到 10 秒")
    parser.add_argument("--ref_text", default="", help="参考音频的文本，留空则自动识别")
    parser.add_argument("--ref_lang", default="zh", help="参考音频的语言：zh 或 en")
    parser.add_argument("--version", default="v2ProPlus")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ref_text = args.ref_text or transcribe(args.ref_audio)
    print(f"ref text: {ref_text}")

    base = "GPT_SoVITS/pretrained_models"
    weights = {
        "v2ProPlus": ("s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
        "v2Pro": ("s1v3.ckpt", "v2Pro/s2Gv2Pro.pth"),
        "v2": ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
               "gsv-v2final-pretrained/s2G2333k.pth"),
    }[args.version]
    config = TTS_Config({"custom": {
        "device": "cuda",
        "is_half": True,
        "version": args.version,
        "t2s_weights_path": f"{base}/{weights[0]}",
        "vits_weights_path": f"{base}/{weights[1]}",
        "bert_base_path": f"{base}/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": f"{base}/chinese-hubert-base",
    }})
    tts = TTS(config)
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"vram after load: {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated")

    def synthesize(text: str) -> tuple[int, np.ndarray, float]:
        """合成一句话，返回采样率、音频数据和耗时秒数。"""
        start = time.perf_counter()
        pieces = list(tts.run({
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": args.ref_audio,
            "prompt_text": ref_text,
            "prompt_lang": args.ref_lang,
            "text_split_method": "cut0",
            "batch_size": 1,
            "parallel_infer": True,
            "seed": 42,
        }))
        elapsed = time.perf_counter() - start
        rate = pieces[0][0]
        return rate, np.concatenate([piece[1] for piece in pieces]), elapsed

    synthesize(SENTENCES[0])  # 预热，不计入结果
    torch.cuda.reset_peak_memory_stats()
    print("idx | chars | audio s | synth s |  RTF")
    total_audio = total_time = 0.0
    for index, text in enumerate(SENTENCES):
        rate, audio, elapsed = synthesize(text)
        seconds = len(audio) / rate
        total_audio += seconds
        total_time += elapsed
        sf.write(os.path.join(args.out_dir, f"{args.version}_{index}.wav"), audio, rate)
        print(f"{index:3d} | {len(text):5d} | {seconds:7.2f} | {elapsed:7.2f} | {elapsed / seconds:4.2f}")
    print(f"total: {total_audio:.1f}s audio in {total_time:.1f}s, RTF {total_time / total_audio:.2f}")
    print(f"peak alloc: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
          f"peak reserved: {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()

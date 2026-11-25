"""VAE 解码微基准：对比不同 PyTorch 版本和优化手段下的解码速度。

MuseTalk 的显卡耗时大头是 SD VAE 解码（32x32 潜变量解码为 256x256 图像）。
本脚本只测这一步，方便快速试验加速方案。

用法::

    python bench_vae.py --vae C:/ai_live/repos/MuseTalk/models/sd-vae --modes baseline,channels_last,compile

依赖：torch、diffusers。
"""

import argparse
import time

import torch
from diffusers import AutoencoderKL


def measure(decode, latents: torch.Tensor, rounds: int) -> float:
    """返回每帧解码耗时（毫秒）。

    参数:
        decode: 接收潜变量、返回图像张量的函数
        latents: 一个批次的潜变量
        rounds: 计时轮数
    """
    with torch.no_grad():
        for _ in range(5):  # 预热
            decode(latents)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(rounds):
            decode(latents)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / (rounds * latents.shape[0]) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True, help="sd-vae 模型目录")
    parser.add_argument("--modes", default="baseline,channels_last,compile")
    parser.add_argument("--batch_sizes", default="4,8")
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()

    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    for mode in args.modes.split(","):
        vae = AutoencoderKL.from_pretrained(args.vae).half().cuda().eval()
        decode = lambda z: vae.decode(z).sample  # noqa: E731
        try:
            dtype = torch.float16
            torch.backends.cudnn.benchmark = mode == "cudnn_benchmark"
            if mode == "fp32":
                vae = vae.float()
                dtype = torch.float32
            elif mode == "channels_last":
                vae = vae.to(memory_format=torch.channels_last)
            elif mode == "compile":
                vae.decoder = torch.compile(vae.decoder)
            elif mode == "compile_max":
                vae.decoder = torch.compile(vae.decoder, mode="max-autotune")
            for batch_size in [int(v) for v in args.batch_sizes.split(",")]:
                latents = torch.randn(batch_size, 4, 32, 32, device="cuda", dtype=dtype)
                if mode == "channels_last":
                    latents = latents.to(memory_format=torch.channels_last)
                torch.cuda.reset_peak_memory_stats()
                per_frame = measure(decode, latents, args.rounds)
                print(f"{mode:14s} batch {batch_size}: {per_frame:6.2f} ms/frame, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
        except Exception as error:  # 某种优化在当前环境不可用时，继续测其余模式
            print(f"{mode:14s} failed: {type(error).__name__}: {str(error)[:200]}", flush=True)
        del vae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

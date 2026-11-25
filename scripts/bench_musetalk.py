"""MuseTalk 1.5 实时推理基准测试。

只加载运行时需要的模型（UNet、VAE、Whisper），不加载素材预处理用的
人脸检测和 DWPose，这样测到的显存才是开播时的真实占用。

用法（工作目录必须是 MuseTalk 仓库根目录，素材需已用官方脚本预处理过）::

    python bench_musetalk.py --avatar results/v15/avatars/avator_1 --audio data/audio/eng.wav

依赖：MuseTalk 仓库自带的环境（torch 2.0.1、diffusers、transformers、opencv）。
"""

import argparse
import glob
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from transformers import WhisperModel  # noqa: E402

from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.blending import get_image_blending  # noqa: E402
from musetalk.utils.utils import datagen, load_all_model  # noqa: E402


def load_avatar(avatar_dir: str) -> dict:
    """读取官方脚本预处理好的素材缓存。

    参数:
        avatar_dir: 素材目录，包含 latents.pt、coords.pkl、mask_coords.pkl、full_imgs、mask
    返回:
        含潜变量、人脸框、原始帧、遮罩及遮罩坐标的字典
    抛出:
        FileNotFoundError: 目录不完整时
    """
    def read_sorted(pattern: str) -> list[np.ndarray]:
        paths = sorted(glob.glob(pattern), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        return [cv2.imread(p) for p in paths]

    with open(os.path.join(avatar_dir, "coords.pkl"), "rb") as f:
        coords = pickle.load(f)
    with open(os.path.join(avatar_dir, "mask_coords.pkl"), "rb") as f:
        mask_coords = pickle.load(f)
    return {
        "latents": torch.load(os.path.join(avatar_dir, "latents.pt")),
        "coords": coords,
        "mask_coords": mask_coords,
        "frames": read_sorted(os.path.join(avatar_dir, "full_imgs", "*.png")),
        "masks": read_sorted(os.path.join(avatar_dir, "mask", "*.png")),
    }


def blend_frame(avatar: dict, index: int, face: np.ndarray) -> np.ndarray:
    """把生成的人脸贴回原始帧。

    参数:
        avatar: load_avatar 的返回值
        index: 帧序号（按素材长度循环）
        face: 模型输出的人脸图像
    返回:
        合成后的整帧图像
    """
    count = len(avatar["coords"])
    x1, y1, x2, y2 = avatar["coords"][index % count]
    face = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
    frame = avatar["frames"][index % count].copy()
    return get_image_blending(frame, face, (x1, y1, x2, y2), avatar["masks"][index % count],
                              avatar["mask_coords"][index % count])


def blend_frame_fast(avatar: dict, index: int, face: np.ndarray) -> np.ndarray:
    """只在人脸裁剪区域内做融合的快速版本，结果应与 blend_frame 基本一致。

    官方实现把整帧在 numpy 和 PIL 之间来回转换三次；这里只复制一次整帧，
    融合运算限制在裁剪框内。

    参数与返回值同 blend_frame。
    """
    count = len(avatar["coords"])
    index %= count
    x1, y1, x2, y2 = avatar["coords"][index]
    xs, ys, xe, ye = avatar["mask_coords"][index]
    frame = avatar["frames"][index].copy()
    height, width = frame.shape[:2]
    # 裁剪框可能超出画面，先收进画面范围
    cx0, cy0, cx1, cy1 = max(xs, 0), max(ys, 0), min(xe, width), min(ye, height)
    region = frame[cy0:cy1, cx0:cx1]
    face_large = region.copy()
    face_large[y1 - cy0:y2 - cy0, x1 - cx0:x2 - cx0] = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
    mask = avatar["masks_gray"][index][cy0 - ys:cy1 - ys, cx0 - xs:cx1 - xs, None].astype(np.uint16)
    region[:] = ((face_large.astype(np.uint16) * mask + region.astype(np.uint16) * (255 - mask) + 127) // 255
                 ).astype(np.uint8)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar", required=True, help="预处理好的素材目录")
    parser.add_argument("--audio", required=True, help="测试音频")
    parser.add_argument("--batch_sizes", default="4,8,16,20", help="逗号分隔的批大小")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max_frames", type=int, default=600, help="每个批大小最多测多少帧")
    parser.add_argument("--channels_last", action="store_true", help="模型使用 channels_last 内存布局")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="开启 cudnn.benchmark")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    vae, unet, pe = load_all_model(
        unet_model_path="models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="models/musetalkV15/musetalk.json",
        device=device,
    )
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    timesteps = torch.tensor([0], device=device)
    dtype = unet.model.dtype

    audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
    whisper = WhisperModel.from_pretrained("models/whisper").to(device=device, dtype=dtype).eval()
    avatar = load_avatar(args.avatar)
    avatar["masks_gray"] = [mask[:, :, 0] for mask in avatar["masks"]]
    if args.channels_last:
        unet.model = unet.model.to(memory_format=torch.channels_last)
        vae.vae = vae.vae.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    start = time.perf_counter()
    features, length = audio_processor.get_audio_feature(args.audio, weight_dtype=dtype)
    chunks = audio_processor.get_whisper_chunk(features, device, dtype, whisper, length, fps=args.fps,
                                               audio_padding_length_left=2, audio_padding_length_right=2)
    chunks = chunks[: args.max_frames]
    audio_seconds = time.perf_counter() - start
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"audio feature: {audio_seconds:.2f}s for {len(chunks)} frames")
    print(f"vram after load: {torch.cuda.memory_allocated() / 2**20:.0f} MiB allocated")
    print(f"frame size: {avatar['frames'][0].shape[1]}x{avatar['frames'][0].shape[0]}")
    print("batch | unet ms/f | vae gpu ms/f | to cpu ms/f | blend ms/f | fast blend ms/f | max diff | "
          "fps (fast blend) | peak alloc MiB | peak reserved MiB")

    for batch_size in [int(v) for v in args.batch_sizes.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        unet_time = vae_time = cpu_time = blend_time = fast_time = 0.0
        max_diff = 0
        frame_index = 0
        with torch.no_grad():
            for round_index in range(2):  # 第一轮预热，不计时
                measured = round_index == 1
                frame_index = 0
                for whisper_batch, latent_batch in datagen(chunks, avatar["latents"], batch_size):
                    t0 = time.perf_counter()
                    audio_feature = pe(whisper_batch.to(device))
                    latent_batch = latent_batch.to(device=device, dtype=dtype)
                    pred = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature).sample
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    # 与 vae.decode_latents 等价，但把显卡计算和拷回内存分开计时，
                    # 并在显卡上完成取整和通道翻转
                    image = vae.vae.decode((pred / vae.scaling_factor).to(vae.vae.dtype)).sample
                    image = ((image / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
                    image = image.permute(0, 2, 3, 1).flip(-1)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    faces = image.cpu().numpy()
                    t3 = time.perf_counter()
                    base = frame_index
                    references = [blend_frame(avatar, base + i, face) for i, face in enumerate(faces)]
                    t4 = time.perf_counter()
                    fast = [blend_frame_fast(avatar, base + i, face) for i, face in enumerate(faces)]
                    t5 = time.perf_counter()
                    frame_index += len(faces)
                    if measured:
                        unet_time += t1 - t0
                        vae_time += t2 - t1
                        cpu_time += t3 - t2
                        blend_time += t4 - t3
                        fast_time += t5 - t4
                        diff = np.abs(references[0].astype(np.int16) - fast[0].astype(np.int16)).max()
                        max_diff = max(max_diff, int(diff))
                if frame_index == 0:
                    raise RuntimeError("没有生成任何帧，请检查素材和音频")
        frames = frame_index
        fast_fps = frames / (unet_time + vae_time + cpu_time + fast_time)
        print(f"{batch_size:5d} | {unet_time / frames * 1000:9.1f} | {vae_time / frames * 1000:12.1f} | "
              f"{cpu_time / frames * 1000:11.2f} | {blend_time / frames * 1000:10.1f} | "
              f"{fast_time / frames * 1000:15.2f} | {max_diff:8d} | {fast_fps:16.1f} | "
              f"{torch.cuda.max_memory_allocated() / 2**20:14.0f} | {torch.cuda.max_memory_reserved() / 2**20:17.0f}")


if __name__ == "__main__":
    main()

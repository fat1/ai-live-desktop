"""口型驱动：MuseTalk 1.5 的运行时封装。

与官方脚本相比：
- 只加载开播需要的部件（UNet、VAE 解码器、Whisper 编码器），节省显存；
- 可用 torch.compile 加速（VAE 解码约快 1.5 倍）；
- 贴回原图只处理人脸区域，结果与官方实现逐像素一致，速度约为 3 倍。

素材需先用 MuseTalk 官方脚本预处理（生成 latents.pt、coords.pkl 等）。
依赖：MuseTalk 仓库、torch、diffusers、transformers、opencv。
"""

import glob
import json
import os
import pickle
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


@dataclass
class Avatar:
    """预处理好的人物素材，所有列表按帧对齐、长度相同。"""

    latents: torch.Tensor          # [帧数, 8, 32, 32]，已在显卡上
    coords: list                   # 每帧人脸框 (x1, y1, x2, y2)
    frames: list[np.ndarray]       # 原始帧，BGR
    masks: list[np.ndarray]        # 融合遮罩，单通道
    mask_coords: list              # 每帧遮罩裁剪框

    def __len__(self) -> int:
        return len(self.frames)


class LipSyncEngine:
    """MuseTalk 推理引擎。"""

    def __init__(self, musetalk_root: str | Path, batch_size: int = 8, use_compile: bool = True,
                 fps: int = 25) -> None:
        """
        参数:
            musetalk_root: MuseTalk 仓库根目录（其下 models 目录放权重）
            batch_size: 每批生成的帧数；越大越快但越占显存，12G 显卡建议不超过 8
            use_compile: 是否用 torch.compile 加速，首次调用会多花几十秒编译
            fps: 输出帧率
        抛出:
            FileNotFoundError: 权重文件缺失
        """
        root = Path(musetalk_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from musetalk.models.unet import PositionalEncoding
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel

        models = root / "models"
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        self.batch_size = batch_size
        self.fps = fps

        with open(models / "musetalkV15" / "musetalk.json", encoding="utf-8") as f:
            unet = UNet2DConditionModel(**json.load(f))
        unet.load_state_dict(torch.load(models / "musetalkV15" / "unet.pth", map_location="cpu"))
        self._unet = unet.to(self.device, self.dtype).eval()

        vae = AutoencoderKL.from_pretrained(str(models / "sd-vae"))
        self._scaling = vae.config.scaling_factor
        vae.encoder = None      # 编码只在素材预处理时需要
        vae.quant_conv = None
        self._vae = vae.to(self.device, self.dtype).eval()

        whisper_encoder = WhisperModel.from_pretrained(str(models / "whisper")).encoder
        self._whisper = SimpleNamespace(encoder=whisper_encoder.to(self.device, self.dtype).eval())
        self._audio = AudioProcessor(feature_extractor_path=str(models / "whisper"))
        self._pe = PositionalEncoding(d_model=384).to(self.device, self.dtype)
        self._timesteps = torch.tensor([0], device=self.device)

        # 各阶段累计耗时（秒），用于性能分析
        self.stats = {"audio_feature": 0.0, "model": 0.0, "blend": 0.0}
        if use_compile:
            self._unet = torch.compile(self._unet)
            self._vae.decoder = torch.compile(self._vae.decoder)
        self._warm_up()

    @torch.no_grad()
    def _warm_up(self) -> None:
        """用假数据跑两遍模型，把编译和首次调用的开销放在加载阶段。"""
        latents = torch.zeros(self.batch_size, 8, 32, 32, device=self.device, dtype=self.dtype)
        audio = torch.zeros(self.batch_size, 50, 384, device=self.device, dtype=self.dtype)
        for _ in range(2):
            predicted = self._unet(latents, self._timesteps, encoder_hidden_states=self._pe(audio)).sample
            self._vae.decode(predicted / self._scaling).sample
        torch.cuda.synchronize()

    def load_avatar(self, avatar_dir: str | Path) -> Avatar:
        """读取预处理好的素材目录。

        抛出:
            FileNotFoundError: 目录不完整
        """
        avatar_dir = Path(avatar_dir)

        def read_sorted(folder: str) -> list[np.ndarray]:
            paths = sorted(glob.glob(str(avatar_dir / folder / "*.png")),
                           key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            return [cv2.imread(p) for p in paths]

        with open(avatar_dir / "coords.pkl", "rb") as f:
            coords = pickle.load(f)
        with open(avatar_dir / "mask_coords.pkl", "rb") as f:
            mask_coords = pickle.load(f)
        latents = torch.cat(torch.load(avatar_dir / "latents.pt", map_location="cpu"))
        return Avatar(
            latents=latents.to(self.device, self.dtype),
            coords=coords,
            frames=read_sorted("full_imgs"),
            masks=[mask[:, :, 0] for mask in read_sorted("mask")],
            mask_coords=mask_coords,
        )

    @torch.no_grad()
    def render(self, avatar: Avatar, audio_path: str | Path, start_index: int = 0) -> Iterator[np.ndarray]:
        """按音频生成说话的画面。

        参数:
            avatar: 人物素材
            audio_path: 语音文件
            start_index: 从素材的第几帧开始，用于和上一段画面衔接
        返回:
            逐帧产出 BGR 整帧图像；帧数为 floor(音频秒数 * fps)
        """
        started = time.perf_counter()
        features, length = self._audio.get_audio_feature(str(audio_path), weight_dtype=self.dtype)
        chunks = self._audio.get_whisper_chunk(features, self.device, self.dtype, self._whisper, length,
                                               fps=self.fps, audio_padding_length_left=2,
                                               audio_padding_length_right=2)
        torch.cuda.synchronize()
        self.stats["audio_feature"] += time.perf_counter() - started
        total = len(chunks)
        for begin in range(0, total, self.batch_size):
            started = time.perf_counter()
            audio_batch = chunks[begin:begin + self.batch_size]
            real = len(audio_batch)
            indices = [(start_index + begin + k) % len(avatar) for k in range(real)]
            latent_batch = avatar.latents[indices]
            if real < self.batch_size:
                # 末尾不足一批时补齐到固定大小，避免 torch.compile 为新形状重新编译
                pad = self.batch_size - real
                audio_batch = torch.cat([audio_batch, audio_batch[-1:].expand(pad, -1, -1)])
                latent_batch = torch.cat([latent_batch, latent_batch[-1:].expand(pad, -1, -1, -1)])
            predicted = self._unet(latent_batch, self._timesteps,
                                   encoder_hidden_states=self._pe(audio_batch)).sample
            image = self._vae.decode(predicted / self._scaling).sample
            image = ((image / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            faces = image.permute(0, 2, 3, 1).flip(-1)[:real].cpu().numpy()  # RGB 转 BGR
            self.stats["model"] += time.perf_counter() - started
            for index, face in zip(indices, faces):
                started = time.perf_counter()
                frame = self._blend(avatar, index, face)
                self.stats["blend"] += time.perf_counter() - started
                yield frame

    @staticmethod
    def idle(avatar: Avatar, start_index: int, count: int) -> Iterator[np.ndarray]:
        """产出不说话时的原始画面，用于句间停顿和待机。"""
        for k in range(count):
            yield avatar.frames[(start_index + k) % len(avatar)]

    @staticmethod
    def _blend(avatar: Avatar, index: int, face: np.ndarray) -> np.ndarray:
        """把生成的人脸贴回原始帧，只在人脸裁剪框内做融合。"""
        x1, y1, x2, y2 = avatar.coords[index]
        xs, ys, xe, ye = avatar.mask_coords[index]
        frame = avatar.frames[index].copy()
        height, width = frame.shape[:2]
        cx0, cy0, cx1, cy1 = max(xs, 0), max(ys, 0), min(xe, width), min(ye, height)
        region = frame[cy0:cy1, cx0:cx1]
        face_large = region.copy()
        face_large[y1 - cy0:y2 - cy0, x1 - cx0:x2 - cx0] = cv2.resize(face, (x2 - x1, y2 - y1))
        mask = avatar.masks[index][cy0 - ys:cy1 - ys, cx0 - xs:cx1 - xs, None].astype(np.uint16)
        region[:] = ((face_large.astype(np.uint16) * mask + region.astype(np.uint16) * (255 - mask) + 127) // 255
                     ).astype(np.uint8)
        return frame

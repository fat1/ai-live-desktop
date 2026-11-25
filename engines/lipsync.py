"""口型驱动：MuseTalk 1.5 的运行时封装。

与官方脚本相比：
- 只加载开播需要的部件（UNet、VAE 解码器、Whisper 编码器），节省显存；
- 可用 torch.compile 加速（VAE 解码约快 1.5 倍）；
- 贴回原图只处理嘴和下巴一带，并做锐化，减少放大造成的模糊；
- 支持按块喂入音频、增量生成画面，语音还没合成完就能开始出口型；
- 不说话时的闭嘴画面在加载素材时一次算好缓存，待机不占显卡。

素材需先用 MuseTalk 官方脚本预处理（生成 latents.pt、coords.pkl 等）。
依赖：MuseTalk 仓库、torch、diffusers、transformers、opencv、librosa。
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
import librosa
import numpy as np
import torch
from torch import nn

from engines.enhance import load_enhancer

WHISPER_RATE = 16000


@dataclass
class Avatar:
    """预处理好的人物素材，所有列表按帧对齐、长度相同。"""

    latents: torch.Tensor          # [帧数, 8, 32, 32]，已在显卡上
    coords: list                   # 每帧人脸框 (x1, y1, x2, y2)
    frames: list[np.ndarray]       # 原始帧，BGR
    masks: list[np.ndarray]        # 融合遮罩，单通道
    mask_coords: list              # 每帧遮罩裁剪框
    mask_bounds: list              # 每帧遮罩里非零区域的外接矩形 (x, y, 宽, 高)，裁剪框坐标
    sharpen: float = 0.0           # 锐化强度；有 lip_weights 时只作用在嘴唇附近，否则作用在整个生成区域
    closed_faces: list[np.ndarray] | None = None  # 每帧对应的闭嘴人脸，用于待机
    details: list[np.ndarray] | None = None  # 每帧人脸框内的原始皮肤纹理（高频细节），int8
    lip_weights: list[np.ndarray] | None = None  # 每帧人脸框内的嘴唇权重图（即没有叠纹理的区域），uint8

    def __len__(self) -> int:
        return len(self.frames)


def _mouth_mask(mask: np.ndarray, face_box: tuple, crop_box: tuple) -> np.ndarray:
    """把融合遮罩收窄到嘴和下巴一带。

    参数:
        mask: MuseTalk 预处理生成的遮罩（覆盖整个下半张脸），大小等于裁剪框
        face_box: 人脸框 (x1, y1, x2, y2)，整帧坐标
        crop_box: 遮罩对应的裁剪框，整帧坐标
    返回:
        收窄后的遮罩，边缘羽化
    """
    x1, y1, x2, y2 = face_box
    width, height = x2 - x1, y2 - y1
    # 嘴的中心大约在人脸框高度的 80% 处；椭圆从鼻子下方盖到下巴，左右留出脸颊
    center = (round((x1 + x2) / 2 - crop_box[0]), round(y1 + 0.80 * height - crop_box[1]))
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, center, (round(0.34 * width), round(0.22 * height)), 0, 0, 360, 255, -1)
    ellipse = cv2.GaussianBlur(ellipse, (0, 0), 0.04 * width)
    return (mask.astype(np.uint16) * ellipse // 255).astype(np.uint8)


def _lip_weight(width: int, height: int) -> np.ndarray:
    """人脸框内的嘴唇权重图：嘴唇附近为 1，其余为 0，边缘羽化。

    返回:
        [高, 宽] 的 float32 数组
    """
    weight = np.zeros((height, width), np.float32)
    cv2.ellipse(weight, (width // 2, round(0.78 * height)), (round(0.20 * width), round(0.06 * height)),
                0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(weight, (0, 0), 8.0)


def _skin_detail(frame: np.ndarray, face_box: tuple, lip_weight: np.ndarray) -> np.ndarray:
    """提取人脸框内的皮肤纹理：原图减去它的模糊版本，嘴唇附近置零。

    参数:
        lip_weight: _lip_weight 的返回值
    返回:
        与人脸框同样大小的 int8 数组，可直接加到生成的人脸上
    """
    x1, y1, x2, y2 = face_box
    box = frame[y1:y2, x1:x2].astype(np.float32)
    # 模糊半径决定叠回多粗的细节：2 只有毛孔，7 左右能带回胡茬和细纹（对比图见技术方案）
    detail = box - cv2.GaussianBlur(box, (0, 0), 7.0)
    return np.clip(detail * (1 - lip_weight[:, :, None]), -127, 127).astype(np.int8)


def _to_bgr(image: torch.Tensor) -> np.ndarray:
    """把 [批, 3, 高, 宽]、取值 0 到 1 的 RGB 张量转成 [批, 高, 宽, 3] 的 uint8 BGR 数组。"""
    return (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).flip(-1).cpu().numpy()


class LipSyncEngine:
    """MuseTalk 推理引擎。"""

    def __init__(self, musetalk_root: str | Path, batch_size: int = 8, use_compile: bool = True,
                 fps: int = 25, enhance_weights: str | Path | None = None) -> None:
        """
        参数:
            musetalk_root: MuseTalk 仓库根目录（其下 models 目录放权重）
            batch_size: 每批生成的帧数；越大越快但越占显存，12G 显卡建议不超过 8
            use_compile: 是否用 torch.compile 加速，首次调用会多花几十秒编译
            fps: 输出帧率
            enhance_weights: 嘴部超分网络的权重（见 enhance.py）；不给则不做超分
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

        self._enhancer = load_enhancer(enhance_weights, self.device, self.dtype) if enhance_weights else None

        # 各阶段累计耗时（秒），用于性能分析
        self.stats = {"audio_feature": 0.0, "model": 0.0}
        if use_compile:
            self._unet = torch.compile(self._unet)
            self._vae.decoder = torch.compile(self._vae.decoder)
        self._warm_up()

    @torch.no_grad()
    def _warm_up(self) -> None:
        """用假数据把模型和音频特征通路各跑一遍，把编译和首次调用的开销放在加载阶段。"""
        latents = torch.zeros(self.batch_size, 8, 32, 32, device=self.device, dtype=self.dtype)
        # 重采样和特征提取的首次调用都很慢（各约 2 秒），在这里先走一遍
        librosa.resample(np.zeros(32000, dtype=np.float32), orig_sr=32000, target_sr=WHISPER_RATE)
        features = self._features(np.zeros(WHISPER_RATE, dtype=np.float32))
        audio = features[:1].expand(self.batch_size, -1, -1)
        for _ in range(2):
            predicted = self._unet(latents, self._timesteps, encoder_hidden_states=self._pe(audio)).sample
            self._vae.decode(predicted / self._scaling).sample
        torch.cuda.synchronize()
        self.stats = {"audio_feature": 0.0, "model": 0.0}

    @torch.no_grad()
    def _features(self, audio: np.ndarray) -> torch.Tensor:
        """计算每一帧画面对应的音频特征。

        参数:
            audio: 16 kHz 单声道 float32 音频
        返回:
            [帧数, 50, 384] 的张量，帧数为 floor(音频秒数 * fps)
        """
        started = time.perf_counter()
        segment = 30 * WHISPER_RATE  # Whisper 一次处理 30 秒
        mels = [self._audio.feature_extractor(audio[i:i + segment], return_tensors="pt",
                                              sampling_rate=WHISPER_RATE).input_features.to(dtype=self.dtype)
                for i in range(0, max(len(audio), 1), segment)]
        chunks = self._audio.get_whisper_chunk(mels, self.device, self.dtype, self._whisper, len(audio),
                                               fps=self.fps, audio_padding_length_left=2,
                                               audio_padding_length_right=2)
        self.stats["audio_feature"] += time.perf_counter() - started
        return chunks

    @torch.no_grad()
    def _face_shape(self, face_box: tuple) -> tuple[int, int, int]:
        """_generate 为某一帧产出的人脸图像形状 (高, 宽, 3)。"""
        x1, y1, x2, y2 = face_box
        return (y2 - y1, x2 - x1, 3) if self._enhancer is not None else (256, 256, 3)

    @torch.no_grad()
    def _generate(self, avatar: Avatar, indices: list[int], audio_batch: torch.Tensor) -> list[np.ndarray]:
        """生成一批人脸。不足一批时补齐到固定大小再裁掉，避免 torch.compile 为新形状重新编译。

        返回:
            每帧一张 BGR 人脸。有超分时已缩放到该帧人脸框的大小，blend 直接使用；
            没有超分时是模型原始的 256x256，由 blend 放大
        """
        started = time.perf_counter()
        real = len(indices)
        latent_batch = avatar.latents[indices]
        if real < self.batch_size:
            pad = self.batch_size - real
            audio_batch = torch.cat([audio_batch, audio_batch[-1:].expand(pad, -1, -1)])
            latent_batch = torch.cat([latent_batch, latent_batch[-1:].expand(pad, -1, -1, -1)])
        predicted = self._unet(latent_batch, self._timesteps, encoder_hidden_states=self._pe(audio_batch)).sample
        image = (self._vae.decode(predicted / self._scaling).sample / 2 + 0.5).clamp(0, 1)[:real]
        if self._enhancer is None:
            faces = list(_to_bgr(image))
        else:
            # 下半张脸（嘴所在的一半）做 4 倍超分，上半张脸直接放大 4 倍，拼成 1024x1024。
            # 上半张脸最终不会被用到（融合遮罩只覆盖嘴和下巴），不值得花算力
            half = image.shape[2] // 2
            lower = self._enhancer(image[:, :, half:]).clamp(0, 1)
            upper = nn.functional.interpolate(image[:, :, :half], scale_factor=4, mode="bilinear")
            image = torch.cat([upper, lower], dim=2).float()
            # 从 4 倍结果一次缩放到每帧人脸框的大小。中间不再经过固定的 512x512，
            # 少一次重采样；人脸框大于 512 时也不会先缩小再放大
            faces = []
            for position, index in enumerate(indices):
                height, width, _ = self._face_shape(avatar.coords[index])
                fitted = nn.functional.interpolate(image[position:position + 1], size=(height, width),
                                                   mode="bicubic", antialias=True)
                faces.append(_to_bgr(fitted)[0])
        self.stats["model"] += time.perf_counter() - started
        return faces

    def load_avatar(self, avatar_dir: str | Path, mouth_only: bool = True, sharpen: float = 0.6,
                    closed_mouth: bool = False, skin_detail: bool = True) -> Avatar:
        """读取预处理好的素材目录。

        参数:
            avatar_dir: MuseTalk 预处理输出的素材目录
            mouth_only: 只替换嘴和下巴一带，脸颊和鼻子保留原始画面。模型输出只有 256x256，
                贴回大脸时要放大，替换范围越大，模糊的区域越明显
            sharpen: 锐化强度，0 表示不锐化。skin_detail 开启时只锐化嘴唇附近（那里不叠纹理，
                否则是唯一没有任何增强的区域）；关闭时锐化整个生成区域
            skin_detail: 把原始帧里的皮肤纹理叠回生成区域。生成结果缺少毛孔、胡茬这类细节，
                而这些细节近似随机噪声，位置差一点看不出来，直接取原图的即可。
                嘴唇附近不叠（原图那里是闭着的嘴缝，叠上去会出现重影）
            closed_mouth: 为每一帧准备闭嘴的人脸，供待机时使用。素材视频里的人如果一直在说话，
                待机直接播原始帧会出现"没声音嘴却在动"。首次计算后缓存在素材目录里
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
        masks = [mask[:, :, 0] for mask in read_sorted("mask")]
        if mouth_only:
            masks = [_mouth_mask(mask, box, crop) for mask, box, crop in zip(masks, coords, mask_coords)]
        avatar = Avatar(
            latents=latents.to(self.device, self.dtype),
            coords=coords,
            frames=read_sorted("full_imgs"),
            masks=masks,
            mask_coords=mask_coords,
            sharpen=sharpen,
            mask_bounds=[cv2.boundingRect(mask) for mask in masks],
        )
        if skin_detail:
            lips = [_lip_weight(x2 - x1, y2 - y1) for x1, y1, x2, y2 in coords]
            avatar.details = [_skin_detail(frame, box, lip) for frame, box, lip in zip(avatar.frames, coords, lips)]
            avatar.lip_weights = [np.round(lip * 255).astype(np.uint8) for lip in lips]
        if closed_mouth:
            # 每帧人脸大小可能不同，缓存成一个拉平的一维数组，读取时按各帧的形状切开。
            # 旧格式的缓存长度对不上，会自动重算并覆盖
            cache = avatar_dir / ("closed_faces_sr.npy" if self._enhancer is not None else "closed_faces.npy")
            shapes = [self._face_shape(box) for box in coords]
            ends = np.cumsum([height * width * channels for height, width, channels in shapes])
            if cache.exists() and len(flat := np.load(cache)) == ends[-1]:
                avatar.closed_faces = [flat[end - np.prod(shape):end].reshape(shape)
                                       for end, shape in zip(ends, shapes)]
            else:
                avatar.closed_faces = self._closed_faces(avatar)
                np.save(cache, np.concatenate([face.ravel() for face in avatar.closed_faces]))
        return avatar

    def _closed_faces(self, avatar: Avatar) -> list[np.ndarray]:
        """用静音驱动，算出素材每一帧对应的闭嘴人脸。

        静音的音频特征是固定的，输出只取决于素材帧，所以可以一次算完，之后待机不再占用显卡。
        """
        silence = self._features(np.zeros(2 * WHISPER_RATE, dtype=np.float32))
        feature = silence[len(silence) // 2:len(silence) // 2 + 1]
        faces = []
        for begin in range(0, len(avatar), self.batch_size):
            indices = list(range(begin, min(begin + self.batch_size, len(avatar))))
            faces.extend(self._generate(avatar, indices, feature.expand(len(indices), -1, -1)))
        return faces

    def stream(self, avatar: Avatar, sample_rate: int, start_index: int = 0) -> "SentenceStream":
        """开始为一句话增量生成画面，见 SentenceStream。"""
        return SentenceStream(self, avatar, sample_rate, start_index)

    def render_faces(self, avatar: Avatar, audio: np.ndarray, sample_rate: int,
                     start_index: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """按一整段音频生成说话的人脸（只做显卡推理，不做融合）。

        参数:
            avatar: 人物素材
            audio: int16 单声道音频
            sample_rate: 音频采样率
            start_index: 从素材的第几帧开始，用于和上一段画面衔接
        返回:
            逐帧产出 (素材帧序号, 人脸图像)，把结果交给 blend 即得到整帧画面
        """
        sentence = self.stream(avatar, sample_rate, start_index)
        sentence.feed(audio, render=False)
        yield from sentence.finish()

    @staticmethod
    def idle(avatar: Avatar, index: int) -> tuple[int, np.ndarray | None]:
        """返回不说话时第 index 帧的内容，格式与 render_faces 的产出一致。

        有闭嘴缓存时返回闭嘴人脸，否则返回 None 表示直接使用原始帧。
        """
        index %= len(avatar)
        return index, None if avatar.closed_faces is None else avatar.closed_faces[index]

    @staticmethod
    def blend(avatar: Avatar, index: int, face: np.ndarray | None) -> np.ndarray:
        """把生成的人脸贴回原始帧，只在人脸裁剪框内做融合。face 为 None 时返回原始帧。"""
        index %= len(avatar)
        if face is None:
            return avatar.frames[index]
        x1, y1, x2, y2 = avatar.coords[index]
        xs, ys = avatar.mask_coords[index][:2]
        frame = avatar.frames[index].copy()
        height, width = frame.shape[:2]
        # 只处理遮罩不为零的那一小块（嘴和下巴），并限制在人脸框和画面之内
        bx, by, bw, bh = avatar.mask_bounds[index]
        fx0, fy0 = max(xs + bx, x1, 0), max(ys + by, y1, 0)
        fx1, fy1 = min(xs + bx + bw, x2, width), min(ys + by + bh, y2, height)
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        # 有超分时人脸已经是人脸框的大小；否则要缩放：放大用 Lanczos（比默认的双线性清晰），缩小用区域平均
        if face.shape[:2] != (y2 - y1, x2 - x1):
            shrink = face.shape[0] >= y2 - y1 and face.shape[1] >= x2 - x1
            face = cv2.resize(face, (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4)
        rows, cols = slice(fy0 - y1, fy1 - y1), slice(fx0 - x1, fx1 - x1)
        part = face[rows, cols].astype(np.float32)
        if avatar.sharpen > 0:
            # 有嘴唇权重图时只锐化嘴唇附近，其余区域的细节由下面叠回的皮肤纹理提供
            strength = avatar.sharpen
            if avatar.lip_weights is not None:
                strength = avatar.lip_weights[index][rows, cols, None] * np.float32(avatar.sharpen / 255)
            part += (part - cv2.GaussianBlur(part, (0, 0), 2.0)) * strength
        if avatar.details is not None:
            # 叠回原始帧的皮肤纹理，见 load_avatar 的 skin_detail 参数
            part += avatar.details[index][rows, cols]
        part = np.clip(part + 0.5, 0, 255).astype(np.uint16)
        region = frame[fy0:fy1, fx0:fx1]
        mask = avatar.masks[index][fy0 - ys:fy1 - ys, fx0 - xs:fx1 - xs, None].astype(np.uint16)
        region[:] = ((part * mask + region.astype(np.uint16) * (255 - mask) + 127) // 255).astype(np.uint8)
        return frame


class SentenceStream:
    """为一句话增量生成说话画面：语音分块到达，到一块算一块。

    用法::

        sentence = engine.stream(avatar, sample_rate, start_index)
        for chunk in audio_chunks:
            for index, face in sentence.feed(chunk): ...
        for index, face in sentence.finish(): ...

    每来一块都对"到目前为止的整句音频"重新提取特征（很快），只生成新增的帧。
    最后几帧的特征依赖尚未到达的后续音频，所以先留着，等下一块或 finish 时再算。
    """

    HOLD_BACK = 3  # 特征需要右侧 2 帧上下文，多留 1 帧余量

    def __init__(self, engine: LipSyncEngine, avatar: Avatar, sample_rate: int, start_index: int) -> None:
        self._engine = engine
        self._avatar = avatar
        self._rate = sample_rate
        self._start = start_index
        self._audio = np.zeros(0, dtype=np.int16)
        self.frames_done = 0  # 已生成的帧数；finish 之后即为这句话的总帧数

    def feed(self, chunk: np.ndarray, render: bool = True) -> list[tuple[int, np.ndarray]]:
        """喂入一块 int16 音频，返回这次新生成的 (素材帧序号, 人脸图像)。

        render 为 False 时只累积音频不生成，适合一次性给完整句再 finish。
        """
        self._audio = np.concatenate([self._audio, chunk])
        return self._render(final=False) if render else []

    def finish(self) -> list[tuple[int, np.ndarray]]:
        """这句话的音频已全部喂完，生成剩余的帧。"""
        return self._render(final=True)

    def _render(self, final: bool) -> list[tuple[int, np.ndarray]]:
        engine = self._engine
        audio = librosa.resample(self._audio.astype(np.float32) / 32768, orig_sr=self._rate, target_sr=WHISPER_RATE)
        if final:
            # 句尾补静音到整帧，再多补 10 毫秒抵消取整误差，这样尾音不会因为帧数向下取整而丢掉
            per_frame = WHISPER_RATE / engine.fps
            target = round(np.ceil(len(audio) / per_frame) * per_frame) + WHISPER_RATE // 100
            audio = np.pad(audio, (0, target - len(audio)))
        features = engine._features(audio)
        available = len(features) if final else max(0, len(features) - self.HOLD_BACK)
        produced = []
        while self.frames_done < available:
            take = min(engine.batch_size, available - self.frames_done)
            # 不足一批时等下一块音频凑满再算，省掉补齐的浪费；
            # 例外：这句话还一帧都没出（尽快开口），或者已经是最后一批
            if take < engine.batch_size and not final and self.frames_done > 0:
                break
            indices = [(self._start + self.frames_done + k) % len(self._avatar) for k in range(take)]
            faces = engine._generate(self._avatar, indices, features[self.frames_done:self.frames_done + take])
            produced.extend(zip(indices, faces))
            self.frames_done += take
        return produced

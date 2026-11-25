"""嘴部画面增强：小型超分辨率网络。

口型模型只输出 256x256 的人脸，贴回大脸时要放大，嘴唇边缘会发虚。这里用 Real-ESRGAN 的
紧凑版网络（realesr-general-x4v3，约 120 万参数，BSD-3 许可）对下半张脸做超分，
实测每帧约 3 毫秒。它能让嘴唇轮廓变清楚，但会把皮肤抹平，所以要和"叠回原图皮肤纹理"
（见 lipsync.py）配合使用。

依赖：torch；权重文件 realesr-general-x4v3.pth。
"""

from pathlib import Path

import torch
from torch import nn


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN 紧凑版网络，结构与官方权重对应。"""

    def __init__(self, num_feat: int = 64, num_conv: int = 32, upscale: int = 4) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")


def load_enhancer(weights: str | Path, device: torch.device, dtype: torch.dtype) -> SRVGGNetCompact:
    """加载超分网络。

    参数:
        weights: realesr-general-x4v3.pth 的路径
    返回:
        输入输出都是 [批, 3, 高, 宽]、取值 0 到 1 的 RGB 张量，输出尺寸为输入的 4 倍
    抛出:
        FileNotFoundError: 权重文件不存在
    """
    state = torch.load(weights, map_location="cpu")
    model = SRVGGNetCompact()
    model.load_state_dict(state.get("params", state))
    return model.to(device, dtype).eval()

# TODO
# 需要将光照编码在几何空间上才具有明确的几何含义，不能直接打开成2d的patch。需要改进rope机制。

import OpenEXR
import Imath
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

# 读取 EXR 文件
def read_exr_file(file_path: str, width: int, height: int):
    """
    读取 EXR 文件，并返回光照数据
    """
    exr_file = OpenEXR.InputFile(file_path)
    # 获取通道数据
    dw = exr_file.header()['dataWindow']
    red = np.frombuffer(exr_file.channel('R'), dtype=np.float32)
    green = np.frombuffer(exr_file.channel('G'), dtype=np.float32)
    blue = np.frombuffer(exr_file.channel('B'), dtype=np.float32)

    # 重新调整为正确的形状
    red = red.reshape((height, width))
    green = green.reshape((height, width))
    blue = blue.reshape((height, width))

    # 合并通道数据
    light_map = np.stack([red, green, blue], axis=-1)
    
    # 归一化到 [0, 1] 之间
    light_map = torch.tensor(light_map).float() / 255.0
    return light_map

# 从 equirectangular (latlong) 映射到球面方向向量
def latlong_to_dirs(height: int, width: int) -> torch.Tensor:
    """
    Convert equirectangular (latlong) image pixel grid to 3D direction vectors.
    """
    theta = torch.linspace(0, torch.pi, height).view(height, 1)  # [0, π], elevation
    phi = torch.linspace(0, 2 * torch.pi, width).view(1, width)   # [0, 2π], azimuth

    x = torch.sin(theta) * torch.cos(phi)  # [H, W]
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta).expand_as(x)

    dirs = torch.stack([x, y, z], dim=-1)  # [H, W, 3]
    return dirs

# RoPE位置编码
class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        breakpoint()
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # x [1, 8388608, 3]
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (Tensor): [..., N, D] tensor of queries
            k (Tensor): [..., N, D] tensor of keys
            indices (Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed


# 设置模型参数
hidden_size = 64  # 隐藏层大小
batch_size = 1
height, width = 2048, 4096  # Light map 尺寸

# 读取 EXR 文件并将其转换为光照张量
light_map = read_exr_file('assets/blue_photo_studio_4k.exr', width, height)

# 获取从 equirectangular 转换的方向向量
dirs = latlong_to_dirs(height, width)  # [H, W, 3]
dirs = dirs.view(1, height * width, 3)  # [1, H*W, 3]

# 设定 q 和 k 张量
q = light_map.view(1, height * width, 3)  # 假设 q 就是光照图数据，形状为 [B, N, 3]
k = torch.randn(batch_size, 64, 64, 64, hidden_size)  # 体素数据作为 k

# 创建 RoPE 编码器
rope = RotaryPositionEmbedder(hidden_size)

# 对 q 和 k 应用 RoPE 编码
q_embed, k_embed = rope(q, k, dirs)

print("q_embed shape:", q_embed.shape)
print("k_embed shape:", k_embed.shape)




# from modules.attention.modules import RotaryPositionEmbedder
# import torch

# rope = RotaryPositionEmbedder(hidden_size=96, in_channels=3)

# q = torch.load("debug/patchified_envmap.pt")  # [batch, num_tokens, hidden_dim]  [1, 3, 64, 64, 32, 64]
# print("shape q:", q.shape)
# k = torch.load("debug/encoded_voxelized_scene.pt")  # [batch, num_tokens, hidden_dim]  [1, 8, 16, 16, 16]
# print("shape k:", k.shape)
# # coords = torch.randint(0, 32, (2, 10, 3)).float()  # 假设体素坐标
# coords = torch.load("debug/voxelized_scene.pt")  #  [1, 1, 64, 64, 64]
# print("coords shape:", coords.shape)

# # 形状要匹配
# q_embed, k_embed = rope(q, k, indices=coords)
# print(q_embed.shape)

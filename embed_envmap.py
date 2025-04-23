import OpenEXR
import Imath
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

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

height, width = 2048, 4096  # Light map 尺寸

# 读取 EXR 文件并将其转换为光照张量
light_map = read_exr_file('assets/blue_photo_studio_4k.exr', width, height)

# 获取从 equirectangular 转换的方向向量
dirs = latlong_to_dirs(height, width)  # [H, W, 3]
dirs = dirs.view(1, height * width, 3)  # [1, H*W, 3]

print(dirs.shape) # [1, 8388608, 3]
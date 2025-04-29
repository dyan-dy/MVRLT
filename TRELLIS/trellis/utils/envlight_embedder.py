import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Union, List, Any
from pathlib import Path

class EnvLightModulator(nn.Module):
    def __init__(self, feature_dim, envlight_dim=3):
        super().__init__()
        self.conv = nn.Conv2d(envlight_dim, feature_dim, kernel_size=1) # 1x1卷积用于调整通道数
        self.fc = nn.Linear(feature_dim, feature_dim * 2)  # 输出scale和shift

    def forward(self, envlight):
        # envlight: [B, C, H, W]
        B, C, H, W = envlight.shape # [1, 3, 2048, 4096]

        # 调整通道数
        envlight = self.conv(envlight) # [B, feature_dim, H, W] [1, 64, 2048, 4096]

        # 将 envlight 从 [B, C, H, W] 转换为 [B, H*W, C]
        envlight = envlight.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # [1, 8388608, 64]

        # 应用全连接层
        params = self.fc(envlight)  # [B, H*W, 2*feature_dim] [1, 8388608, 128]
        gamma, beta = params.chunk(2, dim=-1)  # each [B, H*W, feature_dim]

        # 将 gamma 和 beta 重塑为 [B, feature_dim, H, W]
        gamma = gamma.reshape(B, -1, H, W)
        beta = beta.reshape(B, -1, H, W)

        # 应用 gamma 和 beta 到 envlight
        envlight = envlight.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        modulated_features = gamma * envlight + beta  # [B, feature_dim, H, W]

        modulated_features = modulated_features[0, :, :, :]

        # # 重塑 modulated_features 为 [B, feature_dim * H, W]
        # modulated_features = modulated_features.reshape(B, -1, W) # [1, 131072, 4096]

        # # 选择宽度维度的中间切片，确保宽度为 1024
        modulated_features = modulated_features[:, :, W // 2 - 512:W // 2 + 512] # [1, 2048, 1024]

        return modulated_features # [1, 2048, 1024]


class EXREmbedder(nn.Module):
    """
    Embedder module that reads a .exr file and returns an embedding tensor.

    Args:
        in_channels (int): Number of channels in EXR (usually 3 or 4).
        embed_dim (int): Dimension of the output embedding.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, env_light_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv(env_light_tensor)
        x = x.view(x.size(0), -1)    # flatten, 保留 batch 维度
        embedding = self.fc(x)
        return embedding

class HDREmbedder(nn.Module):
    """
    Stub embedder for .hdr environment maps.
    Currently unimplemented (pass).
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # TODO: add HDR embedding layers
        pass

    def forward(self, hdr_path: str) -> torch.Tensor:
        # TODO: implement HDR embedding
        raise NotImplementedError("HDR embedding not implemented yet")

class PNGEmbedder(nn.Module):
    """
    Stub embedder for .png/.jpg environment maps.
    Currently unimplemented (pass).
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # TODO: add PNG embedding layers
        pass

    def forward(self, img_path: str) -> torch.Tensor:
        # TODO: implement PNG embedding
        raise NotImplementedError("PNG embedding not implemented yet")


def envlight_embedder(
    env_path: str,
    exr_embedder: EXREmbedder,
    hdr_embedder: HDREmbedder,
    png_embedder: PNGEmbedder
) -> torch.Tensor:
    """
    Dispatches environment light file to the appropriate embedder based on file extension.

    Args:
        env_path (str): Path to the environment map file (.exr, .hdr, .png, etc.).
        exr_embedder (EXREmbedder): Instance to embed .exr files.
        hdr_embedder (HDREmbedder): Instance to embed .hdr files.
        png_embedder (PNGEmbedder): Instance to embed .png/.jpg files.

    Returns:
        torch.Tensor: Embedding of the environment map.
    """
    ext = Path(env_path).suffix.lower()
    if ext == '.exr':
        return exr_embedder(env_path)
    elif ext == '.hdr':
        return hdr_embedder(env_path)
    elif ext in ('.png', '.jpg', '.jpeg'):
        return png_embedder(env_path)
    else:
        raise ValueError(f"Unsupported environment map format: {ext}")

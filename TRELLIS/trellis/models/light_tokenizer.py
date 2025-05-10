import torch
import torch.nn as nn
import torch.nn.functional as F

class LightTokenizer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=8, img_size=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )

        # 初始化位置编码为一个 1xCxhxw 的小图（可以默认 img_size=128）
        self.register_parameter("pos_embed", nn.Parameter(torch.randn(1, embed_dim, img_size // patch_size, img_size // patch_size)))

    def forward(self, light_map: torch.Tensor):
        x = self.encoder(light_map)  # (B, C, H', W')

        # 动态 resize 位置编码以匹配 x 的 spatial 尺寸
        if self.pos_embed.shape[-2:] != x.shape[-2:]:
            pos_embed = F.interpolate(self.pos_embed, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed  # Add positional encoding
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x

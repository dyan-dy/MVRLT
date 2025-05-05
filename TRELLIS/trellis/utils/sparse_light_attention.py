import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLightAttention(nn.Module):
    def __init__(self, feat_dim=8, light_feat_dim=3, coord_dim=3, embed_dim=64, out_dim=8, H=2048, W=4096):
        super().__init__()
        self.H, self.W = H, W

        # 将 3D coords 投影为 2D
        self.coord_proj = nn.Linear(coord_dim, 2)

        # 原始 feature -> query
        self.q_proj = nn.Linear(feat_dim, embed_dim)

        # 光照图像素 -> key & value
        self.k_proj = nn.Linear(light_feat_dim, embed_dim)
        self.v_proj = nn.Linear(light_feat_dim, embed_dim)

        # 最终输出维度映射
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, coords, feats, lightmap):
        """
        coords: [N, 4] - (b, x, y, z)
        feats: [N, C]  - 原始稀疏特征
        lightmap: [1, 3, H, W] - RGB lightmap
        """
        N = coords.size(0)
        device = coords.device
        xyz = coords[:, 1:4].float()  # [N, 3]

        # 1. 将坐标映射到光照图平面
        projected = self.coord_proj(xyz).sigmoid()  # [N, 2] ∈ [0, 1]
        x_img = (projected[:, 0] * (self.W - 1)).clamp(0, self.W - 1)
        y_img = (projected[:, 1] * (self.H - 1)).clamp(0, self.H - 1)

        # 2. 从 lightmap 抽样像素值（使用 grid_sample）
        grid = torch.stack([
            (x_img / (self.W - 1)) * 2 - 1,
            (y_img / (self.H - 1)) * 2 - 1
        ], dim=-1).unsqueeze(1)  # [N, 1, 2]

        sampled_light = F.grid_sample(lightmap, grid.view(1, -1, 1, 2), align_corners=True)
        light_feat = sampled_light.view(1, 3, -1).squeeze(0).T  # [N, 3]

        # 3. 构建 attention 输入
        q = self.q_proj(feats)          # [N, D]
        k = self.k_proj(light_feat)     # [N, D]
        v = self.v_proj(light_feat)     # [N, D]

        # 4. Attention（逐点、无跨位置）
        attn = F.softmax((q * k).sum(-1, keepdim=True) / (q.shape[-1] ** 0.5), dim=0)  # [N, 1]
        context = attn * v  # [N, D]

        # 5. 输出维度映射
        out = self.out_proj(context)  # [N, out_dim]
        return out

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import OpenEXR
import Imath
import os
import json
from safetensors.torch import load_file
import models
from encoder import load_encoder, encoder

# ========== 1. Scene Encoder Loader =============
def load_scene_encoder(config, weights_path):
    encoder_config = config["models"]["encoder"]
    model_class = getattr(models, encoder_config["name"])
    model = model_class(**encoder_config["args"])
    weights = load_file(weights_path)
    model.load_state_dict(weights)
    return model.eval().float()

# ========== 2. Light Map Encoder (2D Conv) =============
class LightMapEncoder(nn.Module):
    def __init__(self, in_channels=4, out_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        feat = self.encoder(x)  # [B, C, H', W']
        B, C, H, W = feat.shape
        return feat.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

# ========== 3. Absolute Positional Embedding =============
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ========== 4. Attention Fusion =============
class SceneLightAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, scene, light):
        fused, _ = self.attn(query=scene, key=light, value=light)
        return fused

# ========== 5. Load EXR RGBA Image =============
def load_exr_rgba_tensor(path, target_size=(128, 256)):
    exr_file = OpenEXR.InputFile(path)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape(height, width) for c in "RGB"]

    img_np = np.stack(channels, axis=0)  # [4, H, W]
    img_tensor = torch.tensor(img_np)
    img_tensor = T.Resize(target_size, antialias=True)(img_tensor)  # [4, H', W']
    return img_tensor.unsqueeze(0)  # [1, 4, H, W]

# ========== 6. Main Pipeline =============
def fuse_scene_and_envmap(voxel_tensor, envmap_path, encoder_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load encoders
    # scene_encoder = load_scene_encoder(config, weights_path).to(device)
    encoder_model = encoder_model.to(device)
    light_encoder = LightMapEncoder(in_channels=4, out_dim=8).to(device)
    pe_scene = PositionalEmbedding(max_len=4096, dim=8).to(device)
    pe_light = PositionalEmbedding(max_len=256, dim=8).to(device)
    fuser = SceneLightAttentionFusion(dim=8).to(device)

    # 2. Process voxel scene
    voxel_tensor = voxel_tensor.to(device)
    scene_feat = encoder_model(voxel_tensor)  # [1, 8, 16, 16, 16]
    scene_tokens = scene_feat.flatten(2).transpose(1, 2)  # [1, 4096, 8]
    scene_tokens = pe_scene(scene_tokens)

    # 3. Process envmap image (EXR)
    env_tensor = load_exr_rgba_tensor(envmap_path).to(device)  # [1, 4, H, W]
    light_tokens = light_encoder(env_tensor)  # [1, L, 8]
    light_tokens = pe_light(light_tokens)

    # 4. Fuse with attention
    fused = fuser(scene_tokens, light_tokens)  # [1, 4096, 8]
    fused_3d = fused.transpose(1, 2).reshape(1, 8, 16, 16, 16)

    return fused_3d

# ========== 7. Usage Example =============
if __name__ == "__main__":

    encoder_model = load_encoder(
        config_path="configs/ss_vae_conv3d_16l8_fp16.json",
        repo_id="JeffreyXiang/TRELLIS-image-large",
        filename="ckpts/ss_enc_conv3d_16l8_fp16.safetensors",
        cache_dir="cache",
        use_fp16=False
    )
    voxel_path = "debug/voxelized_scene.pt"
    envmap_path = "assets/blue_photo_studio_4k.exr"

    voxel_tensor = torch.load(voxel_path)  # [1, 1, 64, 64, 64]

    fused_output = fuse_scene_and_envmap(voxel_tensor, envmap_path, encoder_model)
    print("Fused output shape:", fused_output.shape)  # [1, 8, 16, 16, 16]

    torch.save(fused_output, "debug/fused_scene_and_envmap.pt")


# import torch
# import torch.nn as nn
# from encoder import load_encoder, encoder

# class AbsolutePositionalEncoding3D(nn.Module):
#     def __init__(self, channels, depth, height, width):
#         super().__init__()
#         self.depth = depth
#         self.height = height
#         self.width = width
#         self.channels = channels

#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, channels, depth, height, width)
#         )
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)

#     def forward(self, x):
#         # x: [B, C, D, H, W]
#         return x + self.pos_embed

# class LightMapEncoder(torch.nn.Module):
#     def __init__(self, in_channels=4, out_dim=8):
#         super().__init__()
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(32, out_dim, kernel_size=3, stride=2, padding=1),  # ↓2
#         )

#     def forward(self, x):
#         feat = self.encoder(x)  # [B, C, H', W']
#         B, C, H, W = feat.shape
#         return feat.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

# class SceneLightAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim_out, n_heads):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=n_heads, batch_first=True)
#         self.proj = nn.Linear(dim_q, dim_out)

#     def forward(self, scene_feat, light_feat):
#         """
#         scene_feat: [B, N, C]  # query
#         light_feat: [B, M, C]  # key/value
#         """
#         out, _ = self.attn(query=scene_feat, key=light_feat, value=light_feat)
#         return self.proj(out)

# # encode scene
# model = load_encoder(
#     config_path="configs/ss_vae_conv3d_16l8_fp16.json",
#     repo_id="JeffreyXiang/TRELLIS-image-large",
#     filename="ckpts/ss_enc_conv3d_16l8_fp16.safetensors",
#     cache_dir="cache",
#     use_fp16=False
# )

# input_tensor = torch.load("debug/voxelized_scene.pt")
# scene_feat = encoder(model, input_tensor, save_path="debug/encoded_voxelized_scene.pt") # [1, 8, 16, 16, 16]

# B, C, D, H, W = scene_feat.shape

# # envlight embedding (ape)
# pos_embedder = AbsolutePositionalEncoding3D(C, D, H, W)
# scene_feat_pe = pos_embedder(scene_feat)  # same shape

# scene_feat_flat = scene_feat_pe.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

# light_feat = torch.randn(B, 1024, C)

# attn_fusion = SceneLightAttention(dim_q=C, dim_kv=C, dim_out=C, n_heads=2)
# fused_feat = attn_fusion(scene_feat_flat, light_feat)  # [B, N, C]

# print("fused_feat shape:", fused_feat.shape)  # [B, N, C] [1, 4096, 8]
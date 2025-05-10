import torch
from trellis.models.light_tokenizer import LightTokenizer
from trellis.utils.load_hdr_image import load_hdr_image

# Step 1: 加载 HDR 环境图（支持 .exr / .hdr）
envmap_path = "/root/autodl-tmp/gaodongyu/MVRLT/assets/blue_photo_studio_4k.exr"
hdr_tensor = load_hdr_image(envmap_path)  # (1, 3, H, W)
print(f"[INFO] Loaded HDR image with shape: {hdr_tensor.shape}, dtype: {hdr_tensor.dtype}")

# Step 2: 确保 tensor 和 model 都在同一设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hdr_tensor = hdr_tensor.to(device)

# Step 3: 初始化 LightTokenizer 并移动到设备上
tokenizer = LightTokenizer().to(device)

# Step 4: 检查位置编码维度是否匹配（调试用）
print(f"[DEBUG] Input shape: {hdr_tensor.shape}")
out = tokenizer(hdr_tensor)  # 应该触发 encoder + pos_embed + flatten/token output
print(f"[DEBUG] Tokenizer output shape: {out.shape}")

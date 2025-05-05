import cv2
import numpy as np
import torch
import OpenEXR
import Imath
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Set the GPU device to use. Default is '0'.
import wandb

def read_exr_as_tensor(env_light_path):
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(env_light_path)

    # 获取图像的通道信息
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 定义 FLOAT 类型（32 位浮动数值）
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # 读取 R, G, B 通道数据
    r_str = exr_file.channel('R', FLOAT)
    g_str = exr_file.channel('G', FLOAT)
    b_str = exr_file.channel('B', FLOAT)

    # 将字符串转换为 numpy 数组
    r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
    g = np.frombuffer(g_str, dtype=np.float32).reshape((height, width))
    b = np.frombuffer(b_str, dtype=np.float32).reshape((height, width))

    # 合并 RGB 通道数据
    img = np.stack([r, g, b], axis=-1)

    # 如果需要转换为 0-255 范围，可以将像素值放大
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # 转换为 PyTorch tensor (C, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    return img_tensor

def encode_envmap(envmap_path: str, method='mean', sh_degree=3):
    # env = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)  # (H, W, 3), float32
    env = read_exr_as_tensor(envmap_path)
    assert env is not None, f"Failed to read envmap at {envmap_path}"
    
    if method == 'mean':
        return env.mean(axis=(0, 1))  # (3,)
    elif method == 'sh':
        from sh_encoder import project_to_spherical_harmonics  # 你需实现/引用这个
        return project_to_spherical_harmonics(env, degree=sh_degree).reshape(-1)  # (27,)
    else:
        raise ValueError("Unknown envmap encoding method")

# test code
# wandb.init(project="mvrlt_sh", name="encode_envmap")

result = encode_envmap(
    envmap_path='../assets/blue_photo_studio_4k.exr',
    method='mean',  # or follow 'sh'convention
    sh_degree=2
)

print(result.shape, result.max(), result.min()) # mean = [2048, 4096]; 直接用sh编码特别慢

# wandb.finish()
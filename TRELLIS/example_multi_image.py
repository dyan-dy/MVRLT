import os
print("setting up sparse attnetion")
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
print("setting up sparse backend")
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ['CUDA_VISIBLE_DEVICES'] = '1'           # Set the GPU device to use. Default is '0'.

import cv2
import torch
import numpy as np
import OpenEXR
import Imath
import imageio
from PIL import Image
print("import pipeline")
from trellis.pipelines import TrellisImageTo3DPipeline
print("import utils")
from trellis.utils import render_utils

import wandb

wandb.init(project="mvrlt", name="sampling")

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

# Load a pipeline from a model folder or a Hugging Face model hub.
# print("🚀 loading pipeline")
local_path = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/cache/25e0d31ffbebe4b5a97464dd851910efc3002d96"
# pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline = TrellisImageTo3DPipeline.from_pretrained(local_path)
pipeline.cuda()

# Load an image
# images = [
#     Image.open("assets/example_multi_image/character_1.png"),
#     Image.open("assets/example_multi_image/character_2.png"),
#     Image.open("assets/example_multi_image/character_3.png"),
# ]
image_folder = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/assets/example_multi_image/bear/images"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]

env_light_path = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/assets/exrs/blue_photo_studio_4k.exr"
env_light_tensor = read_exr_as_tensor(env_light_path) # [1, 3, 2048, 4096], 0.~255.
# breakpoint()

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    env_light_tensor, 
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave("output/sample_multi.mp4", video, fps=30)

wandb.finish()
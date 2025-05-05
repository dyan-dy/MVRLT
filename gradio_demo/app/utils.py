import os
import gc
import torch
import numpy as np
from PIL import Image
from datetime import datetime

GRADIO_TMP = "/tmp/gradio_product3d"
os.makedirs(GRADIO_TMP, exist_ok=True)

def clean_up():
    """清理GPU显存与Python垃圾回收"""
    torch.cuda.empty_cache()
    gc.collect()

def ensure_rgb(img: Image.Image) -> Image.Image:
    """确保图像为 RGB 格式"""
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return img

def load_images_from_files(file_objs, convert_mode="RGB"):
    """从文件对象中加载 PIL 图像，统一格式"""
    images = []
    for file in file_objs:
        img = Image.open(file.name)
        if convert_mode:
            img = img.convert(convert_mode)
        images.append(img)
    return images

def resize_images(images, max_size=512):
    """将所有图像缩放为相同尺寸"""
    resized = []
    for img in images:
        if max(img.size) > max_size:
            scale = max_size / max(img.size)
            new_size = tuple(int(x * scale) for x in img.size)
            img = img.resize(new_size, Image.ANTIALIAS)
        resized.append(img)
    return resized

def remove_background_simple(pil_img: Image.Image, threshold=80) -> Image.Image:
    """基于左上角像素的简单背景去除（适合纯色背景）"""
    arr = np.array(pil_img)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    base = arr[0, 0]
    diff = np.abs(arr.astype(np.int32) - base.astype(np.int32)).sum(axis=-1)
    mask = diff > threshold
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.concatenate([arr, alpha[..., None]], axis=-1)
    return Image.fromarray(rgba)

def save_temp_image(img: Image.Image, suffix=".png"):
    """保存图像到临时路径，返回路径"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(GRADIO_TMP, f"tmp_{ts}{suffix}")
    img.save(path)
    return path

def make_grid(images, rows=None, cols=None, resize=256):
    """将图像拼成网格，用于展示多视图图"""
    if rows is None and cols is None:
        cols = len(images)
        rows = 1
    elif rows is None:
        rows = len(images) // cols + int(len(images) % cols != 0)
    elif cols is None:
        cols = len(images) // rows + int(len(images) % rows != 0)
    else:
        assert rows * cols >= len(images)

    if resize:
        images = [img.resize((resize, resize)) for img in images]
    
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        grid.paste(img, box=((idx % cols) * w, (idx // cols) * h))
    return grid

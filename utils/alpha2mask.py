from PIL import Image
import numpy as np
import os

file_folder = "/root/autodl-tmp/gaodongyu/MVRLT/outputs/renders/gt"
mask_folder = "/root/autodl-tmp/gaodongyu/MVRLT/outputs/renders/mask"
if not os.path.exists(mask_folder):
    os.mkdir(mask_folder)


for file_path in os.listdir(file_folder):
    img = Image.open(os.path.join(file_folder, file_path)).convert("RGBA")
    img_array = np.array(img)
    alpha_channel = img_array[:, :, 3]
    mask = ((alpha_channel > 0).astype(np.uint8)) * 255
    mask_img = Image.fromarray(mask, mode="L")
    mask_idx = file_path.split(".")[0]
    mask_img.save(os.path.join(mask_folder, f"{mask_idx}.png"))
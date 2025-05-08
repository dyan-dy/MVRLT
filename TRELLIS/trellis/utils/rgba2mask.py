import os
from PIL import Image

def split_rgba_images(input_folder, output_image_folder, output_mask_folder):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):  # 可根据需要改为 .tiff 等支持 alpha 的格式
            path = os.path.join(input_folder, filename)
            image = Image.open(path)

            if image.mode == 'RGBA':
                rgb = image.convert('RGB')
                alpha = image.getchannel('A')

                rgb.save(os.path.join(output_image_folder, filename))
                alpha.save(os.path.join(output_mask_folder, filename))
                print(f"Processed: {filename}")
            else:
                print(f"Skipped (not RGBA): {filename}")

# 用法示例
split_rgba_images(
    input_folder='/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/datasets/Bear/images',
    output_image_folder='/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/datasets/Bear/images_rgb',
    output_mask_folder='/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/datasets/Bear/images_alpha'
)

import os
import cv2
import numpy as np

# 设置路径
images_dir = '/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/image_datasets/standford_ORB/train'
masks_dir = '/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/image_datasets/standford_ORB/train_mask'
output_dir = '/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/image_datasets/standford_ORB/train_masked'
os.makedirs(output_dir, exist_ok=True)

# 遍历图像
for filename in os.listdir(images_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 构造完整路径
        image_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)  # 假设文件名相同
        output_path = os.path.join(output_dir, filename)

        # 读取图像和 mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"跳过 {filename}，图像或mask不存在")
            continue

        # 确保 mask 是二值的
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 扩展为 3 通道用于乘法
        binary_mask_3ch = cv2.merge([binary_mask] * 3)

        # 应用 mask
        masked_image = cv2.bitwise_and(image, binary_mask_3ch)

        # 保存结果
        cv2.imwrite(output_path, masked_image)

print("处理完成！")

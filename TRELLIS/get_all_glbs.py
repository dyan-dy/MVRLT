import os
import shutil

def search_glb_files(path_folder, save_path):
    
    # 深度搜索给定文件夹中的所有 .glb 文件，并将其复制到指定保存路径。

    # 参数：
    # path_folder (str) 要搜索的根目录路径。
    # save_path (str) 存储 .glb 文件的目标路径。
    
    
    # 确保目标文件夹存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 递归遍历目录中的所有文件
    for root, dirs, files in os.walk(path_folder):
        for file in files:
            # 检查文件扩展名是否为 .glb
            if file.endswith(".glb"):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(save_path, file)

                # 如果目标文件夹已经存在同名文件，则修改文件名
                counter = 1
                while os.path.exists(destination_file):
                    filename, ext = os.path.splitext(file)
                    destination_file = os.path.join("save_path, f{filename}_{counter}{ext}")
                    counter += 1

                # 复制文件到目标路径
                shutil.copy(source_file, destination_file)
                print(f"文件 {file} 复制到 {destination_file}")

# 示例用法
path_folder = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/datasets" # 替换为要搜索的文件夹路径
save_path = "/root/autodl-tmp/gaodongyu/MVRLT/GLB_all"       # 替换为保存 .glb 文件的目标路径

search_glb_files(path_folder, save_path)

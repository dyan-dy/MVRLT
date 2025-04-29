import os

def find_c_folders_with_few_files(root_dir, file_threshold=65):
    result = []

    # 遍历 A/ 下面的所有 B/
    for b_name in os.listdir(root_dir):
        b_path = os.path.join(root_dir, b_name)
        if not os.path.isdir(b_path):
            continue
        
        # 遍历 B/ 下面的所有 C/
        for c_name in os.listdir(b_path):
            c_path = os.path.join(b_path, c_name)
            if not os.path.isdir(c_path):
                continue

            # 统计当前 C/ 下的直接文件数量
            file_count = sum(
                1 for entry in os.listdir(c_path)
                if os.path.isfile(os.path.join(c_path, entry))
            )

            if file_count < file_threshold:
                # 可以存B和C的组合路径，方便识别
                result.append(f"{b_name}/{c_name}")
                print(f"C folder '{b_name}/{c_name}' only has {file_count} files.")

    print(f"\nTotal {len(result)} C folders found with less than {file_threshold} files.")
    return result

# 使用示例
root_dir = 'image_datasets/already_done/rank3'#'image_datasets/HSSD'  # 根目录A
output = find_c_folders_with_few_files(root_dir)

# 保存结果到txt
with open('low_file_count_c_folders.txt', 'w') as f:
    for name in output:
        f.write(name + '\n')

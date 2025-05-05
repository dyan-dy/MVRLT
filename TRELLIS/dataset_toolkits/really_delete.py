import os
import shutil

def batch_delete_dirs(txt_file_path, base_prefix="image_datasets/HSSD/"):
    """
    从指定的 txt 文件中读取目录相对路径列表，
    在每行前拼接 base_prefix，再递归删除对应目录。
    """
    # 读取所有目录相对路径
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        rel_paths = [line.strip() for line in f if line.strip()]
    
    for rel in rel_paths:
        # 拼接前缀路径
        dir_path = os.path.join(base_prefix, rel)
        # 转为绝对路径（根据需要可注释掉）
        abs_path = os.path.abspath(dir_path)
        
        if os.path.isdir(abs_path):
            try:
                shutil.rmtree(abs_path)
                print(f"已删除目录: {abs_path}")
            except Exception as e:
                print(f"删除目录失败: {abs_path}，错误：{e}")
        else:
            print(f"目录不存在，跳过: {abs_path}")
if __name__ == "__main__":
    # 假设要删除的目录列表保存在当前目录下的 to_delete.txt
    batch_delete_dirs("low_file_count_c_folders.txt")

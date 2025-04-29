import os
import json

def collect_b_c_structure(root_dir, json_path):
    bc_dict = {}

    # 遍历 A/ 下的所有 B 目录
    for b_name in os.listdir(root_dir):
        b_path = os.path.join(root_dir, b_name)
        if os.path.isdir(b_path):
            c_list = []
            # 遍历 B/ 下所有 C 目录
            for c_name in os.listdir(b_path):
                c_path = os.path.join(b_path, c_name)
                if os.path.isdir(c_path):
                    c_list.append(c_name)
            if c_list:
                bc_dict[b_name] = sorted(c_list)  # 排序一下C名字列表（可选）
    
    # 保存成 JSON
    with open(json_path, 'w') as f:
        json.dump(bc_dict, f, indent=2)

    print(f"Collected structure for {len(bc_dict)} B folders into {json_path}")


# 使用例子
root_dir = 'image_datasets/already_done'  # 根目录 A
json_path = 'assets/multview_pose/existed_names_1.json'
collect_b_c_structure(root_dir, json_path)

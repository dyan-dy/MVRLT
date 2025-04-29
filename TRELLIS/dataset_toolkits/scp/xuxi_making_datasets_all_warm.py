# infer_toy.py
# Part of code modified from lss3d, hunyuan3d2 and trellis.
import time



import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import trimesh
#from dataset_toolkits.lds import sphere_hammersley_sequence
from utils import sphere_hammersley_sequence
from subprocess import DEVNULL, call
from PIL import Image
from math import exp
from lpips import LPIPS
from pathlib import Path
import json

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def init_camera_pose(num_views):
    import json
    pose_dir = 'assets/multview_pose/pose_' + str(num_views) + '.json'
    if (Path(pose_dir).exists()):
        with open(pose_dir, 'r') as f:
            print("\nAlready exists views: "+ pose_dir)
            return json.load(f)  # already exists views
    else:
        print("generating views")
        yaws = []
        pitchs = []
        offset = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset)
            yaws.append(y)
            pitchs.append(p)
        radius = [2] * num_views
        fov = [40 / 180 * np.pi] * num_views
        views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

        with open(pose_dir, 'w') as f:
            json.dump(views, f, indent=2)
    return views


def blender_render_to_multiview(output_dir, output_type, views,input_dir,envmap_dir):
    output_folder = os.path.join(output_dir, output_type)
    # Setup
    print(f"setting up blender render ....\n\nSaving images in dir:{output_folder}\n\n")
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render_all.py'),
        '--',
        '--views', json.dumps(views),
        '--object', '',  # your inputs
        '--resolution', '1024',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--object_all_dir',input_dir,
        '--envmap_path', envmap_dir
    ]

    debug_file = "image_datasets/warm_bpy_debug.txt"
    with open(debug_file, 'w') as f:
        call(args, stdout=f, stderr=f)

    #if os.path.exists(os.path.join(output_folder, 'transforms.json')):
    return True




# ================ main function ========================
# TODO: Please finish the main function logic by your own.

def main():


    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # initial camera pose
    num_views = 64
    multi_views = init_camera_pose(num_views)
    input_dir = "datasets/HSSD/raw/objects"
    
    output_dir = "image_datasets/HSSD" 
    

    # 遍历目录下所有文件
    # env_folder_path = 'assets/exrs/'
    # for env_name in os.listdir(env_folder_path):
    #     if env_name.endswith('.exr'):
    #         envname_without_ext = os.path.splitext(env_name)[0]
    #         output_type = envname_without_ext

    #         env_path = os.path.join(env_folder_path, env_name)
    #         print(f"使用环境光: {env_path}\n\n")
    #         results = blender_render_to_multiview(output_dir, output_type,  multi_views,input_dir,envmap_dir=env_path)
    env_folder_path = 'assets/exrs/'
    output_type = 'warm_restaurant_night_4k'
    #blue_photo_studio_4k.exr  'brown_photostudio_02_4k'  'industrial_sunset_puresky_4k' 'kloppenheim_06_puresky_4k' 'minedump_flats_4k'
    #resting_place_4k 'rogland_moonlit_night_4k'
    # warm_restaurant_night_4k zwartkops_curve_afternoon_4k
    env_path = os.path.join(env_folder_path, output_type+'.exr')
    results = blender_render_to_multiview(output_dir, output_type,  multi_views,input_dir,envmap_dir=env_path)
if __name__ == "__main__":
    main()

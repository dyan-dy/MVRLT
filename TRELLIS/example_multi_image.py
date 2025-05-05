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
import json
import imageio
from PIL import Image
print("import pipeline")
from trellis.pipelines import TrellisImageTo3DPipeline
print("import utils")
from trellis.utils import render_utils
from trellis.utils.encoder_envmap import read_exr_as_tensor
from trellis.trainers.relit_trainer import RelitTrainer

import wandb

wandb.init(project="mvrlt", name="wo_attention")

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
# image_folder = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/assets/example_multi_image/bear/images"
image_folder = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/image_datasets/Standford_ORB/train_masked"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]

env_light_path = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/assets/exrs/blue_photo_studio_4k.exr"
env_light_tensor = read_exr_as_tensor(env_light_path) # [2048, 4096], 0.~255.
# # breakpoint()

pose_path = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/assets/example_multi_image/bear/images/pose_64_3.json"
with open(pose_path, "r") as f:
    poses = json.load(f)

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

# render 3d gaussian at certain viewpoint to screen for supervision
with torch.enable_grad():
    gs_instance = outputs['gaussian'][0]  # already on cuda
    # relit_trainer = RelitTrainer()
    # relit_trainer(gt_images, gs_instance)
    # relighted_gs = render(gs)
    # compare(relighted_gs, gt_image)
    trainer = RelitTrainer(
        gs=gs_instance,
        gt_images=images,
        poses=poses,
        render_type="default",
        epochs=100,
        lr=1e-2,
        batch_size=1,
        loss_type='mse'
    )
    trainer.train()



# breakpoint()
# video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_gs = render_utils.render_video(gs_instance)['color']
# video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
# video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave("debug/sample_train.mp4", video_gs, fps=30)

wandb.finish()
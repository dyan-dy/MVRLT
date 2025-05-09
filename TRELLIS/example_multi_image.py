import os
print("setting up sparse attnetion")
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
print("setting up sparse backend")
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'           # Set the GPU device to use. Default is '0'.

import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
import OpenEXR
import Imath
import json
import imageio
from PIL import Image
from safetensors.torch import load_file
from trellis import models, datasets, trainers
print("import pipeline")
from trellis.pipelines import TrellisImageTo3DPipeline
print("import utils")
from trellis.utils import render_utils
from trellis.utils.encoder_envmap import read_exr_as_tensor
# from trellis.trainers.relit_trainer import RelitTrainer, PoseGTImageDataset
from trellis.datasets.relit_slat2render import RelitDataset
from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder, ElasticSLatGaussianDecoder
from trellis.models.structured_latent_vae.encoder import ElasticSLatEncoder
from trellis.modules.sparse import SparseTensor
from trellis.trainers.vae.structured_latent_vae_gaussian import SLatVaeGaussianTrainer

import wandb

wandb.init(project="mvrlt", name="debug")

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
    
    # config and load model
    with open("/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json", "r") as f:
        config = json.load(f)

    model_decoder = ElasticSLatGaussianDecoder(**config["models"]["decoder"]["args"]).to('cuda')
    model_encoder = ElasticSLatEncoder(**config["models"]["encoder"]["args"]).to('cuda')
    refine_model_dict = {"encoder": model_encoder, "decoder": model_decoder} # 在这得加载整个flow模型

    checkpoint_path = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/cache/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
    checkpoint = load_file(checkpoint_path, device="cuda")
    model_decoder.load_state_dict(checkpoint)
    print("✅ Successfully loaded .safetensors checkpoint.")

    # prepare dataset
    # dataset = getattr(datasets, config["dataset"]["name"])(image_folder, env_light_tensor, poses, **config["dataset"]["args"])
    # dataset = getattr(datasets, config["dataset"]["name"])(image_size=512, model=model_decoder, resolution=512, min_aesthetic_score=0.1, roots=image_folder) # 这里数据格式和原始的不兼容，而且没有metadata
    # dataset = None
    # data_path = "/root/autodl-tmp/gaodongyu/MVRLT/TRELLIS/image_datasets/Standford_ORB/cup"
    # dataset = RelitDataset(data_path)
    # dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    # 替换成你本地的数据目录
    data_root = 'datasets/relit'  # eg. '/home/user/datasets/myset'
    image_size = 512

    # 实例化 dataset
    print("Creating dataset...")
    dataset = RelitDataset(root_dir=data_root, image_size=image_size)

    # 创建 dataloader
    print("Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=RelitDataset.collate_fn
    )

    # 测试迭代一次
    print("Testing dataloader...")
    # breakpoint()
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Image shape:", batch["image"].shape)
        print("Alpha shape:", batch["alpha"].shape)
        print("Coords shape:", batch["coords"].shape)
        print("Feats shape:", batch["feats"].shape)
        print("Intrinsics:", batch["intrinsics"].shape)
        print("Extrinsics:", batch["extrinsics"].shape)



    # finetune the model
    # breakpoint()
    # trainer = getattr(trainers, 'SLatVaeGaussianTrainer')(refine_model_dict, dataset, **config["trainer"]["args"], output_dir='debug', load_dir=None, step=100000)
    trainer = SLatVaeGaussianTrainer(refine_model_dict, dataset, **config["trainer"]["args"], output_dir='debug', load_dir=None, step=100000)
    # print(trainer)
    trainer.run()


# breakpoint()
video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
# video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
# video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave("debug/sample_train.mp4", video_gs, fps=30)

wandb.finish()
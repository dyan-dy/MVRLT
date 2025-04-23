import os
import json
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import models 

def load_encoder(config_path: str, repo_id: str, filename: str, cache_dir: str = "cache", use_fp16: bool = False):
    # 下载或定位权重
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    print(f"[Cache] Using file: {weights_path}")

    # 读取模型配置
    with open(config_path, "r") as f:
        config = json.load(f)
    encoder_config = config["models"]["encoder"]

    # 初始化模型
    model_class = getattr(models, encoder_config["name"])
    model = model_class(**encoder_config["args"])

    # 加载权重
    weights = load_file(weights_path)
    model.load_state_dict(weights)

    # 设置精度与 eval 模式
    model = model.half() if use_fp16 else model.float()
    model.eval()

    print(f"[Model] Loaded {encoder_config['name']} successfully.")
    return model

def encoder(model, input_tensor: torch.Tensor, save_path: str = None):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if save_path:
            torch.save(output, save_path)
            print(f"[Output] Saved to {save_path}")
        return output


# ====================

# # 配置路径
# config_path = 'configs/ss_vae_conv3d_16l8_fp16.json'
# # weights_url = "JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.safetensors"
# weights_local_path = "ss_enc_conv3d_16l8_fp16.safetensors"

# repo_id = "JeffreyXiang/TRELLIS-image-large"  # 仓库ID
# filename = "ckpts/ss_enc_conv3d_16l8_fp16.safetensors"  # 文件名
# cache_dir = "cache"  # 缓存目录

# weights_local_path = hf_hub_download(
#     repo_id=repo_id, 
#     filename=filename,
#     cache_dir="cache"  # 可以选择设置缓存路径
# )

# print(f"[Cache] Using file: {weights_local_path}")

# # 解析配置文件
# with open(config_path, "r") as file:
#     config = json.load(file)

# encoder_config = config["models"]["encoder"]
# model_class = getattr(models, encoder_config["name"])
# model = model_class(**encoder_config["args"])

# # 加载 safetensors 权重
# weights = load_file(weights_local_path)
# model.load_state_dict(weights)

# # fp16 安全和兼容性设置
# # model.half()
# model = model.float() # 不支持fp16计算 "compute_columns3d" not implemented for 'Half'
# model.eval()

# print(f"[Model] Loaded {encoder_config['name']} from config & weights!")

# # 测试推理
# with torch.no_grad():
#     dummy_input = torch.load("debug/voxelized_scene.pt") # torch.randn(1, 1, 64, 64, 64) 
#     # print(dummy_input.dtype)
#     output = model(dummy_input)
#     torch.save(output, "debug/encoded_voxelized_scene.pt")
#     print("输出 Shape:", output.shape)

# ==========

# load encoder structure

# config_path = 'configs/ss_vae_conv3d_16l8_fp16.json'
# with open(config_path, "r") as file:
#     config = json.load(file)

# encoder_config = config["models"]["encoder"]
# encoder = getattr(models, encoder_config["name"])(**encoder_config["args"])#.cuda()

# print(encoder)

# # load pretrained weights

# @staticmethod
# def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
#     """
#     Load a pretrained model.

#     Args:
#         path (str): The path to the model. Can be either local path or a Hugging Face repository.
#     """
#     pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
#     new_pipeline = TrellisTextTo3DPipeline()
#     new_pipeline.__dict__ = pipeline.__dict__
#     args = pipeline._pretrained_args

#     new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
#     new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

#     new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
#     new_pipeline.slat_sampler_params = args['slat_sampler']['params']

#     new_pipeline.slat_normalization = args['slat_normalization']

#     new_pipeline._init_text_cond_model(args['text_cond_model'])

#     return new_pipeline

# # encode scene


import os
import json
import torch
import numpy as np
import random
from easydict import EasyDict as edict

from trellis import models, datasets, trainers
from trellis.datasets.sparse_feat2relit import SparseFeat2Relit
from trellis.models.structured_latent_vae.encoder import SLatEncoder, ElasticSLatEncoder
from trellis.models.structured_latent_vae.decoder_gs_relit import LightTokenizer, ConditionedSLatGaussianDecoder, ElasticConditionedSLatGaussianDecoder
from trellis.trainers.vae.relit_structured_latent_vae_gaussian import DecoderFinetuneTrainer

import wandb

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# 设定随机种子确保结果可复现
def setup_rng(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(random)

def main():
    # 从固定路径加载配置文件
    config_path = "configs/vae/relit_slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json"
    config = json.load(open(config_path, 'r'))
    cfg = edict(config)
    
    # 设置输出目录
    output_dir = "./outputs/0514"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    setup_rng()
    
    # 初始化wandb
    wandb.init(project="mvrlt", name="simple_train")

    torch.autograd.set_detect_anomaly(True) # 检查梯度链在哪里断裂
    
    # 加载数据集
    print("loading dataset")
    dataset = SparseFeat2Relit("./datasets/Proc_Data", **cfg.dataset.args)
    # breakpoint()

    print("loading model")

    # Replace dynamic loading with direct instantiation
    model_dict = {}

    # Initialize the encoder
    if "encoder" in cfg.models:
        model_config = cfg.models["encoder"]
        model_dict["encoder"] = ElasticSLatEncoder(**model_config.args).cuda()

    # Initialize the decoder
    if "decoder" in cfg.models:
        model_config = cfg.models["decoder"]
        model_dict["decoder"] = ElasticConditionedSLatGaussianDecoder(**model_config.args).cuda()

    # Print model information
    for name, model in model_dict.items():
        print(f"\nModel: {name}")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params}, Trainable: {num_trainable}")
    # breakpoint()
    # 加载断点(如果需要)
    load_ckpt = cfg.get('load_ckpt', None)
    
    # 构建训练器
    print("loading trainer")
    trainer = DecoderFinetuneTrainer(
        model_dict,
        dataset,
        **cfg.trainer.args,
        output_dir=output_dir,
        load_dir=cfg.get('load_dir', output_dir),
        step=load_ckpt
    )
    
    # 开始训练
    print("start training")
    trainer.run()
    
    # 完成训练
    wandb.finish()

if __name__ == '__main__':
    main()
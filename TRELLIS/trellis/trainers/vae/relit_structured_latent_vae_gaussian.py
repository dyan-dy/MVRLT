from typing import Dict, Tuple, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import copy
import wandb

import utils3d.torch
from ..basic import BasicTrainer
from .structured_latent_vae_gaussian import SLatVaeGaussianTrainer
from ...representations import Gaussian
from ...renderers import GaussianRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips

class DecoderFinetuneTrainer(SLatVaeGaussianTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cached_feats = None  # 用来缓存第一次变换结果

        self.latent_projector = torch.nn.Linear(
            in_features=1024,  # 替换成你实际的输入维度
            out_features=768  # 替换成 decoder 需要的维度
        ).to(self.device)

        # 冻结 encoder 所有参数
        if 'encoder' in self.models:
            for p in self.models['encoder'].parameters():
                p.requires_grad = False

            # # 从 training_models 中移除 encoder，避免其出现在 optimizer 参数中
            # if 'encoder' in self.training_models:
            #     del self.training_models['encoder']

        # 确保 decoder 模型的参数能进行训练
        self.training_models = {'encoder':self.models['encoder'], 'decoder': self.models['decoder']}
        self._enable_decoder_gradients()  # 启用 decoder 梯度

    def _enable_decoder_gradients(self):
        """确保 decoder 模型的所有参数都需要梯度"""
        for name, param in self.models['decoder'].named_parameters():
            param.requires_grad = True  # 确保 decoder 的所有参数都有梯度


    def training_losses(
        self,
        feats2: SparseTensor,
        image_light: torch.Tensor,
        image_env: torch.Tensor,
        # alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        使用直接提供的 latent feats1，训练 decoder。
        """
        # breakpoint()
        # print("🏂 feats1 type", type(feats1))
        # z = feats1.feats.requires_grad_()  # feats1 直接作为 latent，不需要通过 encoder

        if self.cached_feats is None:
            if hasattr(feats2, 'feats'):  # 支持 SparseTensor
                raw_feats = feats2.feats
            else:
                raw_feats = feats2
            z = self.latent_projector(raw_feats)
            self.cached_feats = z.detach().clone()  # 缓存第一次变换结果
        else:
            z = self.cached_feats

        # z, mean, logvar = self.training_models['encoder'](feats2, sample_posterior=True, return_raw=True)
        # print("z.shape", z.shape)
        reps = self.training_models['decoder'](z)
        self.renderer.rendering_options.resolution = image_env.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)

        device = image_env.device
        # terms = edict(loss=0.0, rec=0.0)
        terms = edict(
            loss=torch.tensor(0.0, device=device, requires_grad=True),
            rec=torch.tensor(0.0, device=device, requires_grad=True)
        )
        # terms = edict()

        rec_image = render_results['color']
        # gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]
        gt_image = image_env

        # print("terms[loss] requires_grad:", terms["loss"].requires_grad)
        # print("terms[loss] grad_fn:", terms["loss"].grad_fn)
        # breakpoint()
        if self.loss_type == 'l1':
            print("loss_type is l1")
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            print("loss_type is l2")
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if self.lambda_ssim > 0:
            print("lambda ssim", self.lambda_ssim)
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            print("lambda lpips", self.lambda_lpips)
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        terms["loss"] = terms["loss"] + terms["rec"]

        # 不再计算 KL loss
        terms["kl"] = torch.tensor(0.0, device=image_env.device)

        # Regularization
        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss

        status = self._get_status(z, reps)

        # ===== wandb logging (每个 loss 项都记录) =====
        print("✍ recodring loss ... ", terms.items())
        wandb_log = {}
        for key, value in terms.items():
            if isinstance(value, torch.Tensor):
                wandb_log[f"loss/{key}"] = value.item()

        # 记录图像（batch 中第 0 张）
        wandb_log["image/rec"] = wandb.Image(rec_image[0].clamp(0, 1))
        wandb_log["image/gt"] = wandb.Image(gt_image[0].clamp(0, 1))

        wandb.log(wandb_log)

        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}
        return terms, status


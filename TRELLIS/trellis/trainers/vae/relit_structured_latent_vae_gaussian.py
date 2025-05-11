from typing import Dict, Tuple, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import copy

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

        # 冻结 encoder 所有参数
        if 'encoder' in self.models:
            for p in self.models['encoder'].parameters():
                p.requires_grad = False

            # 从 training_models 中移除 encoder，避免其出现在 optimizer 参数中
            if 'encoder' in self.training_models:
                del self.training_models['encoder']

        # 只保留 decoder 作为训练模型
        self.training_models = {'decoder': self.models['decoder']}

    def training_losses(
        self,
        latents: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        使用 freeze 住的 encoder 得到的 latent，训练 decoder。
        """
        # 强制 encoder 推理，但不计算梯度
        with torch.no_grad():
            z = self.models['encoder'](latents, sample_posterior=True, return_raw=False)

        reps = self.training_models['decoder'](z)
        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)

        terms = edict(loss=0.0, rec=0.0)

        rec_image = render_results['color']
        gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]

        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        terms["loss"] = terms["loss"] + terms["rec"]

        # KL loss 不计算（因为 encoder 被 freeze 了，没有训练意义）
        terms["kl"] = torch.tensor(0.0, device=image.device)

        # Regularization
        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss

        status = self._get_status(z, reps)

        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}
        return terms, status

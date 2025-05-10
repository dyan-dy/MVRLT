from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from ...modules.attention import CrossAttentionLayer
from ...utils.random_utils import hammersley_sequence
from .base import SparseTransformerBase
from ...representations import Gaussian
from ..sparse_elastic_mixin import SparseTransformerElasticMixin
from ..structured_latent_vae.decoder_gs import SLatGaussianDecoder


class ConditionedSLatGaussianDecoder(SLatGaussianDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_attn = CrossAttentionLayer(  # 你后续实现
            dim=self.model_channels,
            context_dim=128,  # 怎么更优雅地传参
            num_heads=4,
        )

    def forward(self, x: sp.SparseTensor, light_tokens: Optional[torch.Tensor] = None) -> List[Gaussian]:
        h = super().forward_features(x)  # 只走 transformer，不做 out_layer 和 representation

        # 插入光照引导的 CrossAttention
        if light_tokens is not None:
            h.feats = self.cross_attn(h.feats, context=light_tokens)

        h = self.out_layer(h)
        return self.to_representation(h)
    

class ElasticConditionedSLatGaussianDecoder(SparseTransformerElasticMixin, ConditionedSLatGaussianDecoder):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass

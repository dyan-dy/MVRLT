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
# from .._light_tokenizer import LightTokenizer


# LightTokenizer class remains unchanged
class LightTokenizer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=8, img_size=128, learning_rate=0.001, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size))
            layers.append(nn.ReLU(inplace=True))
            in_channels = embed_dim

        self.encoder = nn.Sequential(*layers)
        
        self.register_parameter("pos_embed", nn.Parameter(torch.randn(1, embed_dim, img_size // patch_size, img_size // patch_size)))
        self.learning_rate = learning_rate

    def forward(self, light_map: torch.Tensor):
        print("✅💡 LightTokenizer forward called")
        x = self.encoder(light_map)
        if self.pos_embed.shape[-2:] != x.shape[-2:]:
            pos_embed = F.interpolate(self.pos_embed, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x

# ConditionedSLatGaussianDecoder with tokenizer integration
class ConditionedSLatGaussianDecoder(SLatGaussianDecoder):
    def __init__(self, *args, tokenizer_config: Optional[dict] = None, **kwargs):
        """
        Initialize the decoder and integrate the tokenizer into the model.
        
        Args:
            tokenizer_config (Optional[dict]): Configuration for the light tokenizer.
        """
        # Initialize the parent class first
        super().__init__(*args, **kwargs)

        # Now handle tokenizer initialization, which requires calling `super().__init__()` first
        if tokenizer_config is not None:
            self.tokenizer = LightTokenizer(
                in_channels=tokenizer_config.get("in_channels", 3),
                embed_dim=tokenizer_config.get("embed_dim", 128),
                patch_size=tokenizer_config.get("patch_size", 8),
                img_size=tokenizer_config.get("img_size", 128),
                learning_rate=tokenizer_config.get("learning_rate", 0.001)
            )
        else:
            self.tokenizer = None

        self.input_proj = nn.Linear(1024, self.model_channels) 

        # Add cross-attention layer for light conditioning
        self.cross_attn = CrossAttentionLayer(
            query_dim=self.model_channels,  # Use the model's channels as the query dimension
            context_dim=128,  # Context dimension from tokenizer
            num_heads=12,
        )
    

    def forward(self, x: sp.SparseTensor, light_map: Optional[torch.Tensor] = None) -> List[Gaussian]:
        """
        Forward pass with light conditioning.
        
        Args:
            x (sp.SparseTensor): Input sparse tensor.
            light_map (Optional[torch.Tensor]): Light map to be used for conditioning.
        
        Returns:
            List[Gaussian]: Output Gaussian distributions.
        """
        print("✅ Decoder forward called")
        # Get features through the transformer
        # h = self.forward_features(x)
        print("🌼 x", x.type, x.coords.shape, x.feats.shape)
        if x.feats.shape[1] != self.model_channels:
            x = sp.SparseTensor(
                feats=self.input_proj(x.feats),  # 对 feature 做 Linear
                coords=x.coords                   # 保留坐标不变
            )
        # h = super().forward(x)
        h = x
        print("🌷 h", h.feats.shape, h.coords.shape)

        # If light map is provided, use the tokenizer to process it
        if light_map is not None and self.tokenizer is not None:
            light_tokens = self.tokenizer(light_map)
            h.feats = self.cross_attn(h.feats, context=light_tokens)
        print("🌻 h", h.feats.shape, h.coords.shape)

        # Pass through the output layer and representation layer
        h = self.out_layer(h)
        print("🍂 h", h.feats.shape, h.coords.shape)
        return self.to_representation(h)

# ElasticConditionedSLatGaussianDecoder
class ElasticConditionedSLatGaussianDecoder(SparseTransformerElasticMixin, ConditionedSLatGaussianDecoder):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
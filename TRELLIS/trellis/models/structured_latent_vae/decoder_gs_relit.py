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
from ...modules import sparse as sp
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
# class ConditionedSLatGaussianDecoder(SLatGaussianDecoder):
#     def __init__(self, *args, tokenizer_config: Optional[dict] = None, **kwargs):
#         """
#         Initialize the decoder and integrate the tokenizer into the model.
        
#         Args:
#             tokenizer_config (Optional[dict]): Configuration for the light tokenizer.
#         """
#         # Initialize the parent class first
#         super().__init__(*args, **kwargs)

#         # Now handle tokenizer initialization, which requires calling `super().__init__()` first
#         if tokenizer_config is not None:
#             self.tokenizer = LightTokenizer(
#                 in_channels=tokenizer_config.get("in_channels", 3),
#                 embed_dim=tokenizer_config.get("embed_dim", 128),
#                 patch_size=tokenizer_config.get("patch_size", 8),
#                 img_size=tokenizer_config.get("img_size", 128),
#                 learning_rate=tokenizer_config.get("learning_rate", 0.001)
#             )
#         else:
#             self.tokenizer = None
        
#         # print("self.model_channels: ", self.model_channels)
#         # self.input_proj = nn.Linear(1024, self.model_channels)
#         #  
#         # self.model_channels = 768  # 你最终 transformer 的维度
#         # self.input_layer = sp.SparseLinear(in_features=1024, out_features=self.model_channels)

#         # Add cross-attention layer for light conditioning
#         self.cross_attn = CrossAttentionLayer(
#             query_dim=self.model_channels,  # Use the model's channels as the query dimension
#             context_dim=128,  # Context dimension from tokenizer
#             num_heads=12,
#         )
    

#     def forward(self, x: sp.SparseTensor, light_map: Optional[torch.Tensor] = None) -> List[Gaussian]:
#         """
#         Forward pass with light conditioning.
        
#         Args:
#             x (sp.SparseTensor): Input sparse tensor.
#             light_map (Optional[torch.Tensor]): Light map to be used for conditioning.
        
#         Returns:
#             List[Gaussian]: Output Gaussian distributions.
#         """
#         print("✅ Decoder forward called")
#         # Get features through the transformer
#         # h = self.forward_features(x)
        
#         # if x.feats.shape[1] != self.model_channels:
#         #     x = sp.SparseTensor(
#         #         feats=self.input_proj(x.feats).requires_grad_(),  # 对 feature 做 Linear
#         #         coords=x.coords                   # 保留坐标不变
#         #     )
#         # else:
#         x = sp.SparseTensor(
#             feats=x.feats.requires_grad_(),  # 对 feature 做 Linear
#             coords=x.coords                   # 保留坐标不变
#         )
        
#         print("🌼 x", x.type, x.coords.shape, x.feats.shape)
#         print("🧪 x.feats.requires_grad:", x.feats.requires_grad)
#         print("🧪 x.feats.grad_fn:", x.feats.grad_fn)
#         h = super().forward(x)
#         # h = x
#         print("🌷 h", h.feats.shape, h.coords.shape)
#         print("🧪 x.feats.requires_grad:", h.feats.requires_grad)
#         print("🧪 x.feats.grad_fn:", h.feats.grad_fn)

#         # If light map is provided, use the tokenizer to process it
#         if light_map is not None and self.tokenizer is not None:
#             light_tokens = self.tokenizer(light_map).requires_grad_()
#             h.feats = self.cross_attn(h.feats, context=light_tokens).requires_grad_()
#         print("🌻 h", h.feats.shape, h.coords.shape)
#         print("🧪 x.feats.requires_grad:", h.feats.requires_grad)
#         print("🧪 x.feats.grad_fn:", h.feats.grad_fn)

#         # Pass through the output layer and representation layer
#         h = self.out_layer(h)
#         print("🍂 h", h.feats.shape, h.coords.shape)
#         print("🧪 x.feats.requires_grad:", h.feats.requires_grad)
#         print("🧪 x.feats.grad_fn:", h.feats.grad_fn)
#         return self.to_representation(h)



class ConditionedSLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        cross_attn_layer: int,
        tokenizer_config: Optional[dict] = None,
        context_dim: int = 128,
        representation_config: dict = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            **kwargs
        )

        self.resolution = resolution
        self.cross_attn_layer = cross_attn_layer
        self.rep_config = representation_config

        self.cross_attn = CrossAttentionLayer(
            query_dim=model_channels,
            context_dim=context_dim,
            num_heads=kwargs.get("num_heads", 8),
        )

        self.tokenizer = LightTokenizer(**tokenizer_config) if tokenizer_config else None
        self.out_layer = sp.SparseLinear(model_channels, self._calc_output_channels())
        self._build_perturbation()
        self.initialize_weights()

        if kwargs.get("use_fp16", False):
            self.convert_to_fp16()

    def _calc_output_channels(self):
        layout = {
            '_xyz': {'size': self.rep_config['num_gaussians'] * 3},
            '_features_dc': {'size': self.rep_config['num_gaussians'] * 3},
            '_scaling': {'size': self.rep_config['num_gaussians'] * 3},
            '_rotation': {'size': self.rep_config['num_gaussians'] * 4},
            '_opacity': {'size': self.rep_config['num_gaussians']},
        }
        self.layout = {}
        start = 0
        for k, v in layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
            self.layout[k] = v
        return start

    def _build_perturbation(self):
        pert = hammersley_sequence(3, 0, self.rep_config['num_gaussians'])
        pert = torch.tensor(pert).float() * 2 - 1
        pert = pert / self.rep_config['voxel_size']
        pert = torch.atanh(pert)
        self.register_buffer('offset_perturbation', pert)
    
    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=0,
                aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
                mininum_kernel_size = self.rep_config['3d_filter_kernel_size'],
                scaling_bias = self.rep_config['scaling_bias'],
                opacity_bias = self.rep_config['opacity_bias'],
                scaling_activation = self.rep_config['scaling_activation']
            )
            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            print("🐱 xyz offset", xyz.shape)
            print("🐱 x ", x.coords.shape, x.layout)
            for k, v in self.layout.items():
                if k == '_xyz':
                    offset = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape'])
                    offset = offset * self.rep_config['lr'][k]
                    if self.rep_config['perturb_offset']:
                        offset = offset + self.offset_perturbation
                    offset = torch.tanh(offset) / self.resolution * 0.5 * self.rep_config['voxel_size']
                    print("🦊 xyz offset", xyz.shape, offset.shape)
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = x.feats[x.layout[i]][:, v['range'][0]:v['range'][1]].reshape(-1, *v['shape']).flatten(0, 1)
                    feats = feats * self.rep_config['lr'][k]
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret


    def forward(self, x: sp.SparseTensor, light_map: Optional[torch.Tensor] = None) -> List[Gaussian]:
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:])
        h = h.type(self.dtype)

        if light_map is not None and self.tokenizer is not None:
            context = self.tokenizer(light_map)  # (B, N, C)
        else:
            context = None

        for i, block in enumerate(self.blocks):
            h = block(h)
        
        return self.to_representation(h)

# ElasticConditionedSLatGaussianDecoder
class ElasticConditionedSLatGaussianDecoder(SparseTransformerElasticMixin, ConditionedSLatGaussianDecoder):
    """
    Slat VAE Gaussian decoder with elastic memory management.
    Used for training with low VRAM.
    """
    pass
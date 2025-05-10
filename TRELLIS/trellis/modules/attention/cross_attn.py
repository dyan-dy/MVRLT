import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=context_dim,
            vdim=context_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query_feats, context_feats):
        # query_feats: (B, M, C), context_feats: (B, N, D)
        attn_out, _ = self.attn(query_feats, context_feats, context_feats)
        return self.norm(query_feats + attn_out)

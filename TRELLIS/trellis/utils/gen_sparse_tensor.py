# import torch
# import spconv.pytorch as spconv  # 假设你用的是 spconv 的 SparseTensor

# def generate_random_sparse_tensor(num_points=1000, feat_dim=8, spatial_shape=(64, 64, 64), device='cuda'):
#     """
#     随机生成一个 SparseTensor。
    
#     Args:
#         num_points (int): 点的数量
#         feat_dim (int): 每个点的特征维度
#         spatial_shape (tuple): 稀疏体素网格的空间尺寸
#         device (str): 'cuda' 或 'cpu'

#     Returns:
#         spconv.SparseTensor: 随机稀疏张量
#     """
#     # 随机生成坐标 (batch_idx, z, y, x)
#     batch_idx = torch.zeros(num_points, 1, dtype=torch.int32)
#     coords = torch.randint(0, spatial_shape[0], (num_points, 3), dtype=torch.int32)
#     coords = torch.cat([batch_idx, coords], dim=1).to(device)

#     # 随机生成特征
#     feats = torch.randn(num_points, feat_dim).to(device)

#     # 创建 SparseTensor
#     st = spconv.SparseTensor(feats, coords, spatial_shape=spatial_shape, device=device)
#     return st
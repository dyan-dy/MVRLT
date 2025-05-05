import torch
from scipy.special import sph_harm

def get_envmap_directions(H, W, device='cuda'):
    """生成每个像素在单位球上的方向向量"""
    theta = torch.linspace(0, torch.pi, H, device=device)  # elevation
    phi = torch.linspace(0, 2 * torch.pi, W, device=device)  # azimuth
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    return theta, phi

def project_to_spherical_harmonics(envmap: torch.Tensor, degree: int = 3, device='cuda') -> torch.Tensor:
    """
    将 envmap 投影到球谐函数上，返回 shape: (3, (degree+1)^2)
    
    Args:
        envmap: torch.Tensor, shape (Batch_num, 3, H, W), HDR 环境贴图
        degree: 球谐最大阶数（通常 2-3）
        device: 计算设备, 默认 'cuda'
        
    Returns:
        sh_coeffs: torch.Tensor, shape (Batch_num, 3, (degree+1)^2)
    """
    
    Batch_num, C, H, W = envmap.shape
    assert C == 3, "Only RGB envmaps supported"

    # 降低分辨率，选择新的 H 和 W
    new_H, new_W = 512, 1024  # 选择合适的分辨率
    envmap_resized = torch.nn.functional.interpolate(envmap, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # 获取单位球面方向
    theta, phi = get_envmap_directions(H, W, device=device)  # shape (H, W)

    # 计算权重（球面采样面积）
    dtheta = torch.pi / H
    dphi = 2 * torch.pi / W
    sin_theta = torch.sin(theta)
    weights = sin_theta * dtheta * dphi  # shape (H, W)

    # 计算球谐基函数
    sh_basis = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            Y_lm = torch.tensor(sph_harm(m, l, phi.cpu().numpy(), theta.cpu().numpy()).real, device=device)
            sh_basis.append(Y_lm)

    sh_basis = torch.stack(sh_basis, dim=-1)  # (H, W, B)
    B = sh_basis.shape[-1]

    # 初始化 SH 系数
    coeffs = torch.zeros((Batch_num, 3, B), dtype=torch.float32, device=device)

    for c in range(3):  # RGB 三个通道
        for b in range(B):
            coeffs[:, c, b] = torch.sum(envmap_resized[:, c, :, :] * sh_basis[..., b] * weights, dim=(1, 2))

    return coeffs  # shape (Batch_num, 3, B)

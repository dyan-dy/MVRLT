# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
# import wandb

# def run_feature_visualization(features):
#     # 创建一个包含 3 个子图的面板
#     fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

#     # 1. 可视化体素占位点（坐标）
#     coords = features['slat_sampler_input_coords'][:, 1:].cpu().numpy()  # (N, 3)
#     axs[0].scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, c='blue')
#     axs[0].set_xlabel('X')
#     axs[0].set_ylabel('Y')
#     axs[0].set_zlabel('Z')
#     axs[0].set_title('3D Voxel Occupancy Visualization')

#     # 2. 使用 PCA 可视化输入特征
#     input_feats = features['slat_sampler_input_feats'].cpu().numpy()  # (N, 8)
#     pca = PCA(n_components=3)
#     input_feats_rgb = pca.fit_transform(input_feats)  # (N, 3)
#     input_feats_rgb -= input_feats_rgb.min()
#     input_feats_rgb /= input_feats_rgb.max()  # 归一化到 [0, 1]

#     axs[1].scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=input_feats_rgb, s=2)
#     axs[1].set_xlabel('X')
#     axs[1].set_ylabel('Y')
#     axs[1].set_zlabel('Z')
#     axs[1].set_title('3D Voxel Features (Input) Visualized via PCA to RGB')

#     # 3. 使用 PCA 可视化输出特征
#     output_feats = features['slat_sampler_output_feats'].cpu().numpy()  # (N, 8)
#     output_feats_rgb = pca.fit_transform(output_feats)  # (N, 3)
#     output_feats_rgb -= output_feats_rgb.min()
#     output_feats_rgb /= output_feats_rgb.max()  # 归一化到 [0, 1]

#     axs[2].scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=output_feats_rgb, s=2)
#     axs[2].set_xlabel('X')
#     axs[2].set_ylabel('Y')
#     axs[2].set_zlabel('Z')
#     axs[2].set_title('3D Voxel Features (Output) Visualized via PCA to RGB')

#     # 调整图像布局
#     plt.subplots_adjust(wspace=0.3)

#     # 4. 将面板图像上传到 WandB
#     wandb.log({"3d_feature_pca_panel": wandb.Image(fig)})

import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import wandb

def run_feature_visualization(features):
    # 1. 可视化体素占位点（坐标）
    coords = features['slat_sampler_input_coords'][:, 1:].cpu().numpy()  # (N, 3)
    fig_voxel = go.Figure()

    # 添加体素占位点
    fig_voxel.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name="Voxel Occupancy"
    ))

    # 更新体素占位点的布局
    fig_voxel.update_layout(
        title="Voxel Occupancy Visualization",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),  # 调整边距
    )

    # 2. 使用 PCA 可视化输入特征
    input_feats = features['slat_sampler_input_feats'].cpu().numpy()  # (N, 8)
    pca = PCA(n_components=3)
    input_feats_rgb = pca.fit_transform(input_feats)  # (N, 3)
    input_feats_rgb -= input_feats_rgb.min()
    input_feats_rgb /= input_feats_rgb.max()  # 归一化到 [0, 1]
    
    fig_input_feats = go.Figure()

    # 输入特征的 3D 可视化
    fig_input_feats.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(size=2, color=input_feats_rgb),
        name="Input Feature Visualization"
    ))

    # 更新布局
    fig_input_feats.update_layout(
        title="Input Feature Visualization",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # 3. 使用 PCA 可视化输出特征
    output_feats = features['slat_sampler_output_feats'].cpu().numpy()  # (N, 8)
    output_feats_rgb = pca.fit_transform(output_feats)  # (N, 3)
    output_feats_rgb -= output_feats_rgb.min()
    output_feats_rgb /= output_feats_rgb.max()  # 归一化到 [0, 1]

    fig_output_feats = go.Figure()

    # 输出特征的 3D 可视化
    fig_output_feats.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(size=2, color=output_feats_rgb),
        name="Output Feature Visualization"
    ))

    # 更新布局
    fig_output_feats.update_layout(
        title="Output Feature Visualization",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # 保存为 HTML 文件
    fig_voxel.write_html("wandb/voxel_occupancy_plot.html")
    fig_input_feats.write_html("wandb/input_feature_plot.html")
    fig_output_feats.write_html("wandb/output_feature_plot.html")

    # 4. 将 HTML 文件上传到 WandB
    wandb.log({
        "voxel_occupancy_plot": wandb.Html("wandb/voxel_occupancy_plot.html"),
        "input_feature_plot": wandb.Html("wandb/input_feature_plot.html"),
        "output_feature_plot": wandb.Html("wandb/output_feature_plot.html")
    })

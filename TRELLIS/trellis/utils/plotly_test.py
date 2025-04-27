import plotly.graph_objects as go
import wandb

# 创建一个简单的 3D 可视化
fig = go.Figure(data=[go.Scatter3d(
    x=[1, 2, 3],
    y=[4, 5, 6],
    z=[7, 8, 9],
    mode='markers'
)])

# 初始化 wandb
wandb.init(project="your_project_name", name="feature_visualization")

# 上传 Plotly 图表到 WandB
wandb.log({"3d_plot": wandb.Plotly(fig)})

# 完成 WandB 运行
wandb.finish()

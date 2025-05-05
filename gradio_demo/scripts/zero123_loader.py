# scripts/zero123_loader.py
def load_zero123_model(ckpt_path, device="cuda"):
    # 实际加载模型逻辑，根据你的 Zero123 实现来替换
    import torch
    from models.zero123 import Zero123Model  # 替换为你的实际类

    model = Zero123Model()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

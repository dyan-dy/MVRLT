import torch
from scripts.zero123_loader import load_zero123_model


class Zero123Runner:
    """
    Zero123 多视图预测模型加载与管理类
    """
    _zero123_model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, ckpt_path="./ckpt/zero123.pth"):
        self.ckpt_path = ckpt_path

    @property
    def model(self):
        if self._zero123_model is None:
            self._zero123_model = load_zero123_model(self.ckpt_path, device=self._device)
        return self._zero123_model

    def init_model(self):
        """预加载模型"""
        _ = self.model

    def reload_model(self):
        """手动刷新"""
        self._zero123_model = None
        self.init_model()


# 默认 runner 实例
zero123_runner = Zero123Runner()

from abc import ABC, abstractmethod
import os
from torch import nn
import torch
from Models.config_mixin import _ConfigMixin
import baseline_config as bl_cf


class _BaseModel(nn.Module, _ConfigMixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @staticmethod
    def load_model(model_path: str = None, suffix: str = None) -> '_BaseModel':
        _model_path = model_path if model_path else os.path.join(
            bl_cf.get_bl_config().output_dir,
            f'model.{suffix}.pth' if suffix else 'model.pth'
        )
        return torch.load(_model_path)

from abc import ABC, abstractmethod
import os
from torch import nn
import torch
from Abstracts.config_mixin import _ConfigMixin
from Utils import cuda


class _BaseModel(nn.Module, _ConfigMixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _ConfigMixin.__init__(self)

    @staticmethod
    def load_model(model_path: str = None, suffix: str = None) -> '_BaseModel':
        _model_path = model_path if model_path else os.path.join(
            _ConfigMixin.get_cf(None).output_dir,
            f'model.{suffix}.pth' if suffix else 'model.pth'
        )
        return torch.load(_model_path, map_location=cuda.get_device(), weights_only=False)

    def write_graph_to_tensorboard(self) -> None:
        self.get_bl_cf().writer.add_graph(
            self,
            input_to_model=torch.randn(
                self.get_bl_cf().training.batch_size,
                self.get_bl_cf().dataset.get_seq_len() if self.get_bl_cf().is_temporal else 1,
                *([12] if self.get_bl_cf().is_joint else []),
                3,
                224,
                224
            )
        )

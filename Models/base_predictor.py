from abc import ABC, abstractmethod
import os

import torch

from Models.config_mixin import _ConfigMixin


class _BasePredictor(_ConfigMixin, ABC):
    def __init__(self, model_path: str = None, suffix: str = None):
        self._model_path = model_path if model_path else os.path.join(
            self.get_bl_cf().output_dir,
            f'model.{suffix}.pth' if suffix else 'model.pth'
        )
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

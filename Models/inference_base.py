from abc import ABC, abstractmethod
import os

from Models.config_mixin import _ConfigMixin


class _InferenceBase(_ConfigMixin, ABC):
    def __init__(self):
        self._model_path = os.path.join(
            self.get_bl_cf().output_dir, 'model.pth')
        self.__load_model()

    @abstractmethod
    def _load_model(self):
        pass

from abc import ABC, abstractmethod
import os
import pickle
from typing import Any

import torch

from Models.config_mixin import _ConfigMixin
from Utils import cuda


class _BaseCheckpoint(_ConfigMixin, ABC):
    def __init__(self, input_path: str = None, epoch=0, **custom_state):
        self._input_path = input_path if input_path else os.path.join(
            self.get_bl_cf().output_dir, 'checkpoint.pth'
        )
        self._output_path = os.path.join(
            self.get_bl_cf().output_dir, 'checkpoint.pth'
        )
        self.epoch = epoch
        self._initial_state = {'epoch': epoch, **custom_state}
        self.update_state(**self._initial_state)

    @abstractmethod
    def update_state(self, **state) -> None:
        pass

    @abstractmethod
    def _get_state_dict(self) -> dict[str, Any]:
        pass

    def reset(self, remove_file=True) -> None:
        self.update_state(**self._initial_state)
        if remove_file and os.path.exists(self._output_path):
            os.remove(self._output_path)

    def save(self):
        torch.save(self._get_state_dict(), self._output_path)

    def load(self, from_input=False) -> None:
        path = self._input_path if from_input else self._output_path
        try:
            state_dict = torch.load(
                path,
                map_location=cuda.get_device()
            )
            self.update_state(**state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No previous '{path}' Checkpoint found, starting fresh.")

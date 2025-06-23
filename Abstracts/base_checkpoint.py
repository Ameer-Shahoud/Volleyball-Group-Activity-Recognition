from abc import ABC, abstractmethod
import os
from typing import Any
import torch
from Abstracts.config_mixin import _ConfigMixin
from Models.metrics import Metrics
from Utils import cuda


class _BaseCheckpoint(_ConfigMixin, ABC):
    def __init__(
        self,
        input_path: str = None,
        suffix: str = None, epoch=0,
        model_state: dict[str, Any] = {},
        best_model_state: dict[str, Any] = {},
        best_model_metrics: Metrics = None,
        **custom_state
    ):
        self._input_path = input_path if input_path else os.path.join(
            self.get_bl_cf().output_dir,
            f'checkpoint.{suffix}.pth' if suffix else 'checkpoint.pth'
        )
        self._output_path = os.path.join(
            self.get_bl_cf().output_dir,
            f'checkpoint.{suffix}.pth' if suffix else 'checkpoint.pth'
        )
        self.epoch = epoch
        self.model_state = model_state
        self.best_model_state = best_model_state
        self.best_model_metrics = best_model_metrics
        self._initial_state = {
            'epoch': epoch,
            'model_state': model_state,
            'best_model_state': best_model_state,
            'best_model_metrics': best_model_metrics,
            **custom_state
        }
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

    def load(self, from_input=False) -> '_BaseCheckpoint':
        path = self._input_path if from_input else self._output_path
        try:
            state_dict = torch.load(
                path,
                map_location=cuda.get_device(),
                weights_only=False,
            )
            self.update_state(**state_dict)
            return self
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No previous '{path}' Checkpoint found, starting fresh.")

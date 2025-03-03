import torch
from Models.inference_base import _InferenceBase
from Models.model_base import ModelBase


class InferenceB1(_InferenceBase):
    def _load_model(self):
        self.model: ModelBase = torch.load(self._model_path)

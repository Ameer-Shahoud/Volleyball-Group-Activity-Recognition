import torch
from Models.base_predictor import _BasePredictor
from Models.base_model import BaseModel


class B1Predictor(_BasePredictor):
    def __init__(self, model_path: str = None):
        super().__init__(model_path)

    def _load_model(self):
        self.model: BaseModel = torch.load(self._model_path)

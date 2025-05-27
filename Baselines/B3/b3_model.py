import torch
from torch import nn
from Baselines.B1.b1_model import B1Model
from Abstracts.base_model import _BaseModel
from Enums.classification_level import ClassificationLevel
from Models.lstm_head import LSTMHead


class B3Model(_BaseModel):
    def __init__(self, pretrained_img_model_path: str):
        super().__init__()

        self.pretrained_img_model: B1Model = _BaseModel.load_model(
            model_path=pretrained_img_model_path,
        )

        for p in self.pretrained_img_model.parameters():
            p.requires_grad = False

        self.lstm_head = LSTMHead(
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.IMAGE)
            )
        )

    def forward(self, x: torch.Tensor):
        _, features = self.pretrained_img_model(x)
        return self.lstm_head(features)

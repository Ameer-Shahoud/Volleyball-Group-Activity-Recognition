import torch
from Baselines.B1.b1_model import B1Model
from Abstracts.base_model import _BaseModel
from Enums.classification_level import ClassificationLevel
from torch import nn


class B4Model(_BaseModel):
    def __init__(self, pretrained_img_model_path: str):
        super().__init__()

        self.pretrained_img_model: B1Model = _BaseModel.load_model(
            model_path=pretrained_img_model_path,
        )

        for p in self.pretrained_img_model.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.classifier = nn.Linear(
            512,
            len(self.get_cf().dataset.get_categories(ClassificationLevel.IMAGE))
        )

    def forward(self, x: torch.Tensor):
        features = self.pretrained_img_model(x, return_features=True)
        lstm_out, _ = self.lstm(features)
        outputs = self.classifier(lstm_out[:, -1, :])
        return outputs

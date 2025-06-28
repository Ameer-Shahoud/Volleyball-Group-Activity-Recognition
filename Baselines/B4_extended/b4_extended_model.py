import torch
from Abstracts.base_model import _BaseModel
from Enums.classification_level import ClassificationLevel
from Modules.backbone import BackboneModel
from torch import nn


class B4ExtendedModel(_BaseModel):
    def __init__(self):
        super().__init__()

        self.img_model = BackboneModel(level=ClassificationLevel.IMAGE)

        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.classifier = nn.Linear(
            512,
            len(self.get_cf().dataset.get_categories(ClassificationLevel.IMAGE))
        )

    def forward(self, x: torch.Tensor):
        _, features = self.img_model(x)
        lstm_out, _ = self.lstm(features)
        outputs = self.classifier(lstm_out[:, -1, :])
        return outputs

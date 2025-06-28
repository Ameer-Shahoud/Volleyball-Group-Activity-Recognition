import torch

from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel
from torch import nn


class B5PlayerModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_model = BackboneModel(level=ClassificationLevel.PLAYER)

        self.lstm = nn.LSTM(2048, 1024, batch_first=True)
        self.classifier = nn.Linear(
            1024,
            len(self.get_cf().dataset.get_categories(ClassificationLevel.PLAYER))
        )

    def forward(self, x: torch.Tensor, return_features=False):
        _, features = self.player_model(x)
        lstm_out, _ = self.lstm(features)
        outputs = self.classifier(lstm_out[:, -1, :])
        return lstm_out if return_features else outputs

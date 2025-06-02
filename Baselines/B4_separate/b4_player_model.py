import torch

from Enums.classification_level import ClassificationLevel
from Models.backbone import BackboneModel
from Abstracts.base_model import _BaseModel
from Models.lstm_head import LSTMHead


class B4PlayerModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_model = BackboneModel(level=ClassificationLevel.PLAYER) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

        self.lstm_head = LSTMHead(
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.PLAYER)
            )
        )

    def forward(self, x: torch.Tensor):
        _, features = self.player_model(x)
        return self.lstm_head(features)

import torch
from torch import nn
from Baselines.B1.b1_model import B1Model
from Abstracts.base_model import _BaseModel
from Enums.classification_level import ClassificationLevel
from Models.backbone import BackboneModel
from Models.lstm_head import LSTMHead


class B3ExtendedModel(_BaseModel):
    def __init__(self):
        super().__init__()

        self.img_model = BackboneModel(level=ClassificationLevel.IMAGE) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

        self.lstm_head = LSTMHead(
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.IMAGE)
            )
        )

    def forward(self, x: torch.Tensor):
        _, features = self.img_model(x)
        return self.lstm_head(features)

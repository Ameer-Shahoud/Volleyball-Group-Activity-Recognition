import torch

from Enums.classification_level import ClassificationLevel
from Models.backbone import BackboneModel
from Models.base_model import _BaseModel


class B1Model(_BaseModel):
    def __init__(self):
        super().__init__()
        self.model = BackboneModel(level=ClassificationLevel.IMAGE) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

    def forward(self, x: torch.Tensor):
        return self.model(x)

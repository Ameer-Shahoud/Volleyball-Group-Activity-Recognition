import torch
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel


class B3PlayerModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.model = BackboneModel(level=ClassificationLevel.PLAYER)
        # .set_backbone_requires_grad(False) \
        # .set_backbone_layer_requires_grad('layer4', True) \
        # .set_backbone_layer_requires_grad('fc', True)

    def forward(self, x: torch.Tensor, return_features=False):
        return self.model(x)[1 if return_features else 0]

import torch
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel


class B1Model(_BaseModel):
    def __init__(self):
        super().__init__()
        self.model = BackboneModel(level=ClassificationLevel.IMAGE)

    def forward(self, x: torch.Tensor, return_features=False):
        return self.model(x)[1 if return_features else 0]

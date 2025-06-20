import torch
from Abstracts.base_model import _BaseModel
from Enums.classification_level import ClassificationLevel
from Modules.backbone import BackboneModel
from Modules.lstm_head import LSTMHead


class B4ExtendedModel(_BaseModel):
    def __init__(self):
        super().__init__()

        self.img_model = BackboneModel(level=ClassificationLevel.IMAGE)
        # .set_backbone_requires_grad(False) \
        # .set_backbone_layer_requires_grad('layer4', True) \
        # .set_backbone_layer_requires_grad('fc', True)

        self.lstm_head = LSTMHead(
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.IMAGE)
            )
        )

    def forward(self, x: torch.Tensor):
        _, features = self.img_model(x)
        return self.lstm_head(features)[0]

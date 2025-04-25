import torch

from Enums.classification_level import ClassificationLevel
from Models.backbone import BackboneModel
from Models.base_model import _BaseModel
from torch import nn

from Models.custom_max_pool import CustomMaxPool


class B2JointModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_base = BackboneModel(level=ClassificationLevel.IMAGE) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

        self.img_head = nn.Sequential(
            CustomMaxPool(dim=1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(
                512,
                len(
                    self.get_cf().dataset.get_categories(ClassificationLevel.IMAGE)
                )
            )
        )

    def forward(self, x: torch.Tensor):
        batch_size, frames_count, players_count, channels, width, height = x.shape
        x_view = x.view(
            batch_size*players_count, frames_count, channels, width, height
        )

        player_outputs, player_features = self.player_base(x_view)

        player_features = player_features.view(batch_size, players_count, -1)
        img_outputs = self.img_head(player_features)

        return player_outputs, img_outputs

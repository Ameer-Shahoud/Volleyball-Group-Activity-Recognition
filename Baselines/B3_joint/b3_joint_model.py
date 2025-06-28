import torch

from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel
from torch import nn

from Modules.custom_max_pool import CustomMaxPool


class B3JointModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_base = BackboneModel(level=ClassificationLevel.PLAYER)

        self.pool = CustomMaxPool(dim=1)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(
                512,
                len(self.get_cf().dataset.get_categories(
                    ClassificationLevel.IMAGE)),
            )
        )

    def forward(self, x: torch.Tensor):
        batch_size, frames_count, players_count, channels, width, height = x.shape
        x_view = x.view(
            batch_size*players_count, frames_count, channels, width, height
        )

        player_outputs, player_features = self.player_base(x_view)

        player_outputs = player_outputs.view(batch_size*players_count, -1)
        player_features = player_features.view(batch_size, players_count, -1)

        img_outputs = self.classifier(self.pool(player_features))

        return player_outputs, img_outputs

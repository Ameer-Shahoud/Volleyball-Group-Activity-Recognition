import torch
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel
from Modules.custom_max_pool import CustomMaxPool
from torch import nn


class B6JointModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_base = BackboneModel(level=ClassificationLevel.PLAYER)\
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer3', True) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True) \
            .freeze_bn_layers()

        self.pool = CustomMaxPool(dim=1)

        self.lstm = nn.LSTM(2048, 1024, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(
                256,
                len(self.get_cf().dataset.get_categories(
                    ClassificationLevel.IMAGE)),
            )
        )

    def forward(self, x: torch.Tensor):
        batch_size, frames_count, players_count, channels, width, height = x.shape

        x = x.transpose(1, 2).contiguous()
        x_view = x.view(
            batch_size*players_count, frames_count, channels, width, height
        )

        player_outputs, player_features = self.player_base(x_view)

        player_outputs = player_outputs.view(
            batch_size*players_count*frames_count, -1
        )

        player_features = player_features.view(
            batch_size, players_count, frames_count, -1
        )

        pooled_features = self.pool(player_features)

        lstm_features, _ = self.lstm(pooled_features)

        total_features = torch.cat(
            [
                pooled_features.view(
                    batch_size, frames_count, 1024, 2).mean(dim=-1),
                lstm_features
            ], dim=2
        )

        img_outputs = self.classifier(total_features[:, -1, :])

        return player_outputs, img_outputs

import torch
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel
from Modules.custom_max_pool import CustomMaxPool
from torch import nn


class B7JointModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_base = BackboneModel(level=ClassificationLevel.PLAYER)\
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer3', True) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

        self.player_lstm = nn.LSTM(2048, 512, batch_first=True)
        self.player_classifier = nn.Linear(
            512,
            len(self.get_cf().dataset.get_categories(ClassificationLevel.PLAYER))
        )

        self.pool = CustomMaxPool(dim=1)

        self.lstm = nn.LSTM(512, 256, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(
                128,
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

        _, player_features = self.player_base(x_view)
        player_temporal_features, _ = self.player_lstm(player_features)
        player_outputs = self.player_classifier(
            player_temporal_features[:, -1, :]
        ).view(batch_size*players_count, -1)

        player_outputs = player_outputs.view(
            batch_size*players_count, -1
        )

        player_temporal_features = player_temporal_features.view(
            batch_size, players_count, frames_count, -1
        )

        pooled_features = self.pool(player_temporal_features)

        temporal_features, _ = self.lstm(pooled_features)

        img_outputs = self.classifier(temporal_features[:, -1, :])

        return player_outputs, img_outputs

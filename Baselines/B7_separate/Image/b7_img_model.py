import torch
from Baselines.B5_separate.Player.b5_player_model import B5PlayerModel
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.custom_max_pool import CustomMaxPool
from torch import nn


class B7ImgModel(_BaseModel):
    def __init__(self, pretrained_player_model_path: str):
        super().__init__()

        self.pretrained_player_model: B5PlayerModel = _BaseModel.load_model(
            model_path=pretrained_player_model_path,
        )

        for p in self.pretrained_player_model.parameters():
            p.requires_grad = False

        # self.layer_norm_1 = nn.LayerNorm(2048)
        # self.layer_norm_2 = nn.LayerNorm(1024)

        self.pool = CustomMaxPool(dim=1)

        self.lstm = nn.LSTM(1024, 512, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            # nn.LayerNorm(256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
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

        _, player_features = self.pretrained_player_model.player_model(x_view)
        # player_features = self.layer_norm_1(player_features)

        player_temporal_features, _ = self.pretrained_player_model.lstm(
            player_features
        )

        player_temporal_features = player_temporal_features.view(
            batch_size, players_count, frames_count, -1)

        # player_temporal_features = self.layer_norm_2(player_temporal_features)

        pooled_features = self.pool(player_temporal_features)

        temporal_features, _ = self.lstm(pooled_features)

        img_outputs = self.classifier(temporal_features[:, -1, :])

        return img_outputs

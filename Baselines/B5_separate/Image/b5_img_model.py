import torch
from Baselines.B5_separate.Player.b5_player_model import B5PlayerModel
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.custom_max_pool import CustomMaxPool
from torch import nn


class B5ImgModel(_BaseModel):
    def __init__(self, pretrained_player_model_path: str):
        super().__init__()

        self.pretrained_player_model: B5PlayerModel = _BaseModel.load_model(
            model_path=pretrained_player_model_path,
        )

        for p in self.pretrained_player_model.parameters():
            p.requires_grad = False

        self.pool = CustomMaxPool(dim=1)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
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

        x = x.transpose(1, 2).contiguous()
        x_view = x.view(
            batch_size*players_count, frames_count, channels, width, height
        )

        _, player_features = self.pretrained_player_model.player_model(x_view)
        player_temporal_features, _ = self.pretrained_player_model.lstm(
            player_features
        )

        total_features = torch.cat(
            [
                player_features[:, -1,
                                :].view(batch_size*players_count, 1024, 2).mean(dim=-1),
                player_temporal_features[:, -1, :]
            ], dim=1
        ).view(batch_size, players_count, -1)

        img_outputs = self.classifier(self.pool(total_features))

        return img_outputs

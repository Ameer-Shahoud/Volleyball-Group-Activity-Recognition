import torch
from Baselines.B3_separate.Player.b3_player_model import B3PlayerModel
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.custom_max_pool import CustomMaxPool
from torch import nn


class B6ImgModel(_BaseModel):
    def __init__(self, pretrained_player_model_path: str):
        super().__init__()

        self.pretrained_player_model: B3PlayerModel = _BaseModel.load_model(
            model_path=pretrained_player_model_path,
        )

        for p in self.pretrained_player_model.parameters():
            p.requires_grad = False

        self.pool = CustomMaxPool(dim=1)

        self.lstm = nn.LSTM(2048, 1024, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
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

        player_features = self.pretrained_player_model(
            x_view, return_features=True
        ).view(
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

        return img_outputs

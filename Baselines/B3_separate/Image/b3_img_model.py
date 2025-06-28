import torch
from Baselines.B3_separate.Player.b3_player_model import B3PlayerModel
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.custom_max_pool import CustomMaxPool
from torch import nn


class B3ImgModel(_BaseModel):
    def __init__(self, pretrained_player_model_path: str):
        super().__init__()

        self.pretrained_player_model: B3PlayerModel = _BaseModel.load_model(
            model_path=pretrained_player_model_path,
        )

        for p in self.pretrained_player_model.parameters():
            p.requires_grad = False

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

        player_features = self.pretrained_player_model(
            x_view, return_features=True
        )

        player_features = player_features.view(batch_size, players_count, -1)
        img_outputs = self.classifier(self.pool(player_features))

        return img_outputs

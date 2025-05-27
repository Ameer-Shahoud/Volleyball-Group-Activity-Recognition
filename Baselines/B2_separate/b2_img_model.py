import torch
from torch import nn
from Baselines.B2_separate.b2_player_model import B2PlayerModel
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Models.custom_max_pool import CustomMaxPool


class B2ImgModel(_BaseModel):
    def __init__(self, pretrained_player_model_path: str):
        super().__init__()

        self.pretrained_player_model: B2PlayerModel = _BaseModel.load_model(
            model_path=pretrained_player_model_path,
        )

        for p in self.pretrained_player_model.parameters():
            p.requires_grad = False

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

        _, player_features = self.pretrained_player_model(x_view)

        player_features = player_features.view(batch_size, players_count, -1)
        img_outputs = self.img_head(player_features)

        return img_outputs

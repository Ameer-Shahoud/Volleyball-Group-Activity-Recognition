import torch
from Baselines.B3_separate.Player.b3_player_model import B3PlayerModel
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.classifier_head import ClassifierHead
from Modules.custom_max_pool import CustomMaxPool
from Modules.lstm_head import LSTMHead
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

        self.lstm = nn.LSTM(2048, 512, batch_first=True)

        self.classifier = ClassifierHead(
            input_dim=512,
            hidden_dim=256,
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.IMAGE)
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

        img_outputs = self.classifier(lstm_features[:, -1, :])

        return img_outputs

    def write_graph_to_tensorboard(self):
        self.get_bl_cf().writer.add_graph(
            self,
            input_to_model=torch.randn(
                self.get_bl_cf().training.batch_size,
                self.get_bl_cf().dataset.get_seq_len(),
                12,
                3,
                224,
                224
            )
        )

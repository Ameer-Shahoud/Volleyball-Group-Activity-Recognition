import torch
from Baselines.B4_separate.b4_player_model import B4PlayerModel
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Models.classifier_head import ClassifierHead


class B4ImgModel(_BaseModel):
    def __init__(self, pretrained_player_model_path: str):
        super().__init__()

        self.pretrained_player_model: B4PlayerModel = _BaseModel.load_model(
            model_path=pretrained_player_model_path,
        )

        for p in self.pretrained_player_model.parameters():
            p.requires_grad = False

        self.img_head = ClassifierHead(
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

        player_features, _ = self.pretrained_player_model(x_view)

        player_features = player_features[
            :, -1, :].view(batch_size, players_count, -1)
        img_outputs = self.img_head(player_features)

        return img_outputs

import torch
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel
from Modules.crops_classifier_head import CropsClassifierHead
from Modules.lstm_head import LSTMHead


class B5JointModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_base = BackboneModel(level=ClassificationLevel.PLAYER)
        # .set_backbone_requires_grad(False) \
        # .set_backbone_layer_requires_grad('layer4', True) \
        # .set_backbone_layer_requires_grad('fc', True)

        self.player_lstm = LSTMHead(
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.PLAYER)
            )
        )

        self.img_head = CropsClassifierHead(
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

        _, player_features = self.player_base(x_view)
        player_outputs, player_features = self.player_lstm(
            player_features.view(batch_size * players_count, frames_count, -1)
        )

        player_features = player_features[
            :, -1, :].view(batch_size, players_count, -1)
        img_outputs = self.img_head(player_features)

        return player_outputs, img_outputs

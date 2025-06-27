import torch

from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel
from Modules.crops_classifier_head import CropsClassifierHead


class B3JointModel(_BaseModel):
    def __init__(self):
        super().__init__()
        self.player_base = BackboneModel(level=ClassificationLevel.PLAYER)
        # .set_backbone_requires_grad(False) \
        # .set_backbone_layer_requires_grad('layer4', True) \
        # .set_backbone_layer_requires_grad('fc', True)

        self.img_head = CropsClassifierHead(
            num_classes=len(self.get_cf().dataset.get_categories(
                ClassificationLevel.IMAGE)
            )
        )

    def forward(self, x: torch.Tensor):
        batch_size, frames_count, players_count, channels, width, height = x.shape
        x_view = x.view(
            batch_size*players_count, frames_count, channels, width, height
        )

        player_outputs, player_features = self.player_base(x_view)

        player_outputs = player_outputs.view(batch_size*players_count, -1)
        player_features = player_features.view(batch_size, players_count, -1)

        img_outputs = self.img_head(player_features)

        return player_outputs, img_outputs

    def write_graph_to_tensorboard(self):
        self.get_bl_cf().writer.add_graph(
            self,
            input_to_model=torch.randn(
                self.get_bl_cf().training.batch_size, 1, 12, 3, 224, 224
            )
        )

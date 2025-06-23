import torch
from Enums.classification_level import ClassificationLevel
from Abstracts.base_model import _BaseModel
from Modules.backbone import BackboneModel
from Utils.cuda import get_device


class B1Model(_BaseModel):
    def __init__(self, write_graph_to_tensorboard=True):
        super().__init__()
        self.model = BackboneModel(level=ClassificationLevel.IMAGE)
        # .set_backbone_requires_grad(False) \
        # .set_backbone_layer_requires_grad('layer4', True) \
        # .set_backbone_layer_requires_grad('fc', True)

        if write_graph_to_tensorboard:
            self._write_graph_to_tensorboard()

    def forward(self, x: torch.Tensor, return_features=False):
        return self.model(x)[1 if return_features else 0]

    def _write_graph_to_tensorboard(self):
        self.to(get_device())
        self.get_bl_cf().writer.add_graph(
            self,
            input_to_model=torch.randn(
                self.get_bl_cf().training.batch_size,
                self.get_bl_cf().dataset.get_seq_len() if self.get_bl_cf().is_temporal else 1,
                3,
                224,
                224
            )
        )

from typing import Iterator
from Baselines.B3_joint.b3_joint_model import B3JointModel
from torch import nn
from Models.joint_loss_trainer import JointLossTrainer


class B3JointTrainer(JointLossTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix
        )

    def _get_model(self) -> B3JointModel:
        return B3JointModel()

    def get_parameters(self) -> tuple[Iterator[nn.Parameter]]:
        model: B3JointModel = self._model
        return model.player_base.parameters(), model.img_head.parameters()

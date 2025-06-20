from typing import Iterator
from Baselines.B5_joint.b5_joint_model import B5JointModel
from torch import nn
from itertools import chain
from Models.joint_loss_trainer import JointLossTrainer


class B5JointTrainer(JointLossTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix
        )

    def _get_model(self) -> B5JointModel:
        return B5JointModel()

    def get_parameters(self) -> tuple[Iterator[nn.Parameter]]:
        model: B5JointModel = self._model
        return chain(model.player_lstm.parameters(), model.player_base.parameters()), model.img_head.parameters()

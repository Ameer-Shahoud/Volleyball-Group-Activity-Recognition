from typing import Iterator
from torch import nn
from itertools import chain

import torch
from Baselines.B6_joint.b6_joint_model import B6JointModel
from Models.joint_loss_trainer import JointLossTrainer


class B6JointTrainer(JointLossTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix
        )

    def _get_model(self) -> B6JointModel:
        return B6JointModel()

    def get_parameters(self) -> tuple[Iterator[nn.Parameter]]:
        model: B6JointModel = self._model
        return model.player_base.parameters(), chain(model.lstm.parameters(), model.classifier.parameters())

    def _transform_player_labels(self, player_labels):
        return torch.repeat_interleave(player_labels, repeats=self.get_bl_cf().dataset.get_seq_len())

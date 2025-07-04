from typing import Iterator
from torch import nn
from itertools import chain

import torch
from Baselines.B7_joint.b7_joint_model import B7JointModel
from Models.joint_loss_trainer import JointLossTrainer


class B7JointTrainer(JointLossTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix
        )

    def _get_model(self) -> B7JointModel:
        return B7JointModel()

    def get_parameters(self) -> tuple[Iterator[nn.Parameter]]:
        model: B7JointModel = self._model
        return (
            chain(model.player_base.parameters(), model.player_lstm.parameters(
            ), model.player_classifier.parameters()),
            chain(model.lstm.parameters(), model.classifier.parameters())
        )

from abc import abstractmethod
from typing import Iterator, Type
import torch
from Abstracts.base_trainer import _BaseTrainer
from Enums.classification_level import ClassificationLevel
from Models.image_players_dataset import ImagePlayersDataset
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from Models.metrics import Metrics


class JointLossTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix,
            loss_levels=[
                ClassificationLevel.PLAYER.value,
                ClassificationLevel.IMAGE.value
            ]
        )

    def _get_dataset_type(self) -> Type[ImagePlayersDataset]:
        return ImagePlayersDataset

    def get_criterions(self) -> list[nn.Module]:
        return [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    def get_optimizers(self) -> list[torch.optim.Optimizer]:
        params = self.get_parameters()
        return [
            optim.AdamW(
                (p for p in params[0] if p.requires_grad),
                lr=self.get_bl_cf().training.learning_rate
            ),
            optim.AdamW(
                (p for p in params[1] if p.requires_grad),
                lr=self.get_bl_cf().training.learning_rate
            )
        ]

    @abstractmethod
    def get_parameters(self) -> tuple[Iterator[nn.Parameter]]:
        pass

    def get_schedulers(self) -> list[lr_scheduler.LRScheduler]:
        config = self.get_bl_cf().training.scheduler
        return [torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizers[0], mode=config.mode, factor=config.factor, patience=config.patience
        ), torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizers[0], mode=config.mode, factor=config.factor, patience=config.patience
        )]

    def _batch_step(self, metrics: Metrics, inputs: torch.Tensor, labels: tuple[torch.Tensor], apply_backward=False):
        player_labels = labels[0]
        img_labels = labels[1]

        player_outputs: torch.Tensor
        img_outputs: torch.Tensor
        player_outputs, img_outputs = self._model(inputs)

        player_labels = self._transform_player_labels(player_labels.view(-1))

        player_loss: torch.Tensor = self._criterions[0](
            player_outputs, player_labels
        )

        img_loss: torch.Tensor = self._criterions[1](img_outputs, img_labels)

        if apply_backward:
            [o.zero_grad() for o in self._optimizers]
            total_loss = 0.3 * player_loss + 0.7 * img_loss
            total_loss.backward()
            [o.step() for o in self._optimizers]

        player_loss = player_loss.item()
        _, player_predicted = player_outputs.max(1)
        metrics.update_metrics(
            level=self._loss_levels[0],
            loss=player_loss,
            predicted=player_predicted,
            labels=player_labels
        )

        img_loss = img_loss.item()
        _, img_predicted = img_outputs.max(1)
        metrics.update_metrics(
            level=self._loss_levels[1],
            loss=img_loss,
            predicted=img_predicted,
            labels=img_labels
        )

    def _transform_player_labels(self, player_labels: torch.Tensor):
        return player_labels

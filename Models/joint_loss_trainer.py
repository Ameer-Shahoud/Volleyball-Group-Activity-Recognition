from abc import abstractmethod
from typing import Iterator, Type
import torch
from Abstracts.base_trainer import _BaseTrainer
from Enums.classification_level import ClassificationLevel
from Models.image_players_dataset import ImagePlayersDataset
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


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
            optim.Adam(
                (p for p in params[0] if p.requires_grad),
                lr=self.get_bl_cf().training.learning_rate
            ),
            optim.Adam(
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

    def _train_batch_step(self, inputs, labels):
        player_loss, player_correct, player_total, img_loss, img_correct, img_total = self._batch_step(
            inputs, labels, apply_backward=True
        )

        self.train_loss[0] += player_loss
        self.train_correct[0] += player_correct
        self.train_total[0] += player_total

        self.train_loss[1] += img_loss
        self.train_correct[1] += img_correct
        self.train_total[1] += img_total

    def _eval_batch_step(self, inputs, labels):
        player_loss, player_correct, player_total, img_loss, img_correct, img_total = self._batch_step(
            inputs, labels
        )

        self.val_loss[0] += player_loss
        self.val_correct[0] += player_correct
        self.val_total[0] += player_total

        self.val_loss[1] += img_loss
        self.val_correct[1] += img_correct
        self.val_total[1] += img_total

    def _test_batch_step(self, inputs, labels):
        player_loss, player_correct, player_total, img_loss, img_correct, img_total = self._batch_step(
            inputs, labels
        )

        self.test_loss[0] += player_loss
        self.test_correct[0] += player_correct
        self.test_total[0] += player_total

        self.test_loss[1] += img_loss
        self.test_correct[1] += img_correct
        self.test_total[1] += img_total

    def _batch_step(self, inputs: torch.Tensor, labels: tuple[torch.Tensor], apply_backward=False) -> tuple:
        player_labels = labels[0]
        img_labels = labels[1]

        player_outputs: torch.Tensor
        img_outputs: torch.Tensor
        player_outputs, img_outputs = self._model(inputs)

        batch_size,  players_count = inputs.shape[0], inputs.shape[2]

        player_outputs = player_outputs.view(batch_size*players_count, -1)
        player_labels = player_labels.view(-1)

        player_loss: torch.Tensor = self._criterions[0](
            player_outputs, player_labels
        ) / players_count

        img_loss: torch.Tensor = self._criterions[1](img_outputs, img_labels)

        if apply_backward:
            total_loss = player_loss + img_loss
            [o.zero_grad() for o in self._optimizers]
            total_loss.backward()
            [o.step() for o in self._optimizers]

        player_loss = player_loss.item()
        _, player_predicted = player_outputs.max(1)
        player_correct = (player_predicted ==
                          player_labels).sum().item()
        player_total = player_labels.size(0)

        img_loss = img_loss.item()
        _, img_predicted = img_outputs.max(1)
        img_correct = (img_predicted == img_labels).sum().item()
        img_total = img_labels.size(0)

        return player_loss, player_correct, player_total, img_loss, img_correct, img_total

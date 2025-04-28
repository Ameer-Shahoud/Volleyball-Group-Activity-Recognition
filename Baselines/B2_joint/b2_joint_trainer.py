from typing import Type
import torch
from Baselines.B2_joint.b2_joint_model import B2JointModel
from Abstracts.base_trainer import _BaseTrainer
from Enums.classification_level import ClassificationLevel
from Models.image_players_dataset import ImagePlayersDataset
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from Utils.cuda import get_device


class B2JointTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            loss_labels=[
                ClassificationLevel.PLAYER.value,
                ClassificationLevel.IMAGE.value
            ]
        )

    def _get_dataset_type(self) -> Type[ImagePlayersDataset]:
        return ImagePlayersDataset

    def _get_model(self) -> B2JointModel:
        return B2JointModel()

    def get_criterions(self) -> list[nn.Module]:
        return [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    def get_optimizers(self) -> list[torch.optim.Optimizer]:
        model: B2JointModel = self._model
        return [optim.Adam(
            (p for p in model.player_base.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        ), optim.Adam(
            model.img_head.parameters(), lr=self.get_bl_cf().training.learning_rate
        )]

    def get_schedulers(self) -> list[lr_scheduler.LRScheduler]:
        return [lr_scheduler.ReduceLROnPlateau(
            self._optimizers[0], mode='min', factor=0.1, patience=2
        ), lr_scheduler.ReduceLROnPlateau(
            self._optimizers[1], mode='min', factor=0.1, patience=2
        )]

    def _train_batch_step(self, inputs, labels):
        player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss = self.__get_outputs(
            inputs, labels)

        total_loss = player_loss + img_loss

        [o.zero_grad() for o in self._optimizers]
        total_loss.backward()
        [o.step() for o in self._optimizers]

        self.train_loss[0] += player_loss.item()
        _, player_predicted = player_outputs.max(1)
        self.train_correct[0] += (player_predicted ==
                                  player_labels).sum().item()
        self.train_total[0] += player_labels.size(0)

        self.train_loss[1] += img_loss.item()
        _, img_predicted = img_outputs.max(1)
        self.train_correct[1] += (img_predicted == img_labels).sum().item()
        self.train_total[1] += img_labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss = self.__get_outputs(
            inputs, labels)

        self.val_loss[0] += player_loss.item()
        _, player_predicted = player_outputs.max(1)
        self.val_correct[0] += (player_predicted ==
                                player_labels).sum().item()
        self.val_total[0] += player_labels.size(0)

        self.val_loss[1] += img_loss.item()
        _, img_predicted = img_outputs.max(1)
        self.val_correct[1] += (img_predicted == img_labels).sum().item()
        self.val_total[1] += img_labels.size(0)

    def _test_batch_step(self, inputs, labels):
        player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss = self.__get_outputs(
            inputs, labels)

        self.test_loss[0] += player_loss.item()
        _, player_predicted = player_outputs.max(1)
        self.test_correct[0] += (player_predicted ==
                                 player_labels).sum().item()
        self.test_total[0] += player_labels.size(0)

        self.test_loss[1] += img_loss.item()
        _, img_predicted = img_outputs.max(1)
        self.test_correct[1] += (img_predicted == img_labels).sum().item()
        self.test_total[1] += img_labels.size(0)

    def __get_outputs(self, inputs, labels):
        player_labels = labels[0]
        img_labels = labels[1]

        player_outputs, img_outputs = self._model(inputs)

        batch_size,  players_count = inputs.shape[0], inputs.shape[2]

        player_outputs = player_outputs.view(batch_size*players_count, -1)
        player_labels = player_labels.view(-1)
        player_loss = self._criterions[0](
            player_outputs, player_labels
        ) / players_count

        img_loss = self._criterions[1](img_outputs, img_labels)

        return player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss

from typing import Type
import torch
from Baselines.B2_joint.b2_joint_history import B2JointHistory, B2JointHistoryItem
from Baselines.B2_joint.b2_joint_model import B2JointModel
from Models.base_trainer import _BaseTrainer
from Models.image_players_dataset import ImagePlayersDataset
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from Utils.cuda import get_device


class B2JointTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(checkpoint_path, history_path)

    def _init_values(self):
        self.player_train_loss, self.player_train_correct, self.player_train_total = 0.0, 0, 0
        self.player_val_loss, self.player_val_correct, self.player_val_total = 0.0, 0, 0
        self.player_test_loss, self.player_test_correct, self.player_test_total = 0.0, 0, 0

        self.img_train_loss, self.img_train_correct, self.img_train_total = 0.0, 0, 0
        self.img_val_loss, self.img_val_correct, self.img_val_total = 0.0, 0, 0
        self.img_test_loss, self.img_test_correct, self.img_test_total = 0.0, 0, 0

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

    def _get_history(self, history_path: str = None) -> B2JointHistory:
        return B2JointHistory(history_path)

    def _on_epoch_step(self, epoch: int):
        super()._on_epoch_step(epoch)

        self._schedulers[0].step(self.player_val_loss)
        self._schedulers[1].step(self.img_val_loss)

        return B2JointHistoryItem(
            epoch,
            self.player_train_loss / self.player_train_loss,
            100 * self.player_train_correct / self.player_train_total,
            self.player_val_loss / self.val_size,
            100 * self.player_val_correct / self.player_val_total,
            self.img_train_loss / self.img_train_loss,
            100 * self.img_train_correct / self.img_train_total,
            self.img_val_loss / self.val_size,
            100 * self.img_val_correct / self.img_val_total,
        )

    def _train_batch_step(self, inputs, labels):
        player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss = self.__get_outputs(
            inputs, labels)

        total_loss = player_loss + img_loss

        [o.zero_grad() for o in self._optimizers]
        total_loss.backward()
        [o.step() for o in self._optimizers]

        self.player_train_loss += player_loss.item()
        _, player_predicted = player_outputs.max(1)
        self.player_train_correct += (player_predicted ==
                                      player_labels).sum().item()
        self.player_train_total += player_labels.size(0)

        self.img_train_loss += img_loss.item()
        _, img_predicted = img_outputs.max(1)
        self.img_train_correct += (img_predicted == img_labels).sum().item()
        self.img_train_total += img_labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss = self.__get_outputs(
            inputs, labels)

        self.player_val_loss += player_loss.item()
        _, player_predicted = player_outputs.max(1)
        self.player_val_correct += (player_predicted ==
                                    player_labels).sum().item()
        self.player_val_total += player_labels.size(0)

        self.img_val_loss += img_loss.item()
        _, img_predicted = img_outputs.max(1)
        self.img_val_correct += (img_predicted == img_labels).sum().item()
        self.img_val_total += img_labels.size(0)

    def _test_batch_step(self, inputs, labels):
        player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss = self.__get_outputs(
            inputs, labels)

        self.player_test_loss += player_loss.item()
        _, player_predicted = player_outputs.max(1)
        self.player_test_correct += (player_predicted ==
                                     player_labels).sum().item()
        self.player_test_total += player_labels.size(0)

        self.img_test_loss += img_loss.item()
        _, img_predicted = img_outputs.max(1)
        self.img_test_correct += (img_predicted == img_labels).sum().item()
        self.img_test_total += img_labels.size(0)

    def _on_test_step(self):
        print(
            f"""Test Results:
            Player Loss: {self.player_test_loss/self.test_size:.4f}, Player Acc: {100 * self.player_test_correct/self.player_test_total:.2f}%
            Image Loss: {self.img_test_loss/self.test_size:.4f}, Image Acc: {100 * self.img_test_correct/self.img_test_total:.2f}%
            """
        )

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

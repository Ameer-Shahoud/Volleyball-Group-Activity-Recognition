from typing import Type
import torch
from Baselines.B2_joint.b2_joint_checkpoint import B2JointCheckpoint
from Baselines.B2_joint.b2_joint_history import B2JointHistory, B2JointHistoryItem
from Models.base_trainer import _BaseTrainer
from Enums.classification_level import ClassificationLevel
from Models.base_model import BaseModel
from Models.custom_max_pool import CustomMaxPool
from Models.image_players_dataset import ImagePlayersDataset
from torch import nn
from torchvision import models
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

    def _prepare_model(self):
        self.player_model = BaseModel(
            backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            level=ClassificationLevel.PLAYER
        ).set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

        self.img_head_model = nn.Sequential(
            CustomMaxPool(dim=1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(
                512,
                len(
                    self.get_cf().dataset.get_categories(ClassificationLevel.IMAGE)
                )
            )
        )

    def _prepare_optimizer(self):
        self.player_criterion = nn.CrossEntropyLoss()
        self.img_criterion = nn.CrossEntropyLoss()
        self.player_optimizer = optim.Adam(
            (p for p in self.player_model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )
        self.img_optimizer = optim.Adam(
            self.img_head_model.parameters(), lr=self.get_bl_cf().training.learning_rate
        )
        self.player_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.player_optimizer, mode='min', factor=0.1, patience=2
        )
        self.img_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.img_optimizer, mode='min', factor=0.1, patience=2
        )

    def _to_available_device(self):
        self.player_model.to_available_device()
        self.img_head_model.to(get_device())

        self.player_criterion.to(get_device())
        self.img_criterion.to(get_device())

        for state in self.player_optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(get_device())
        for state in self.img_optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(get_device())

    def _get_checkpoint(self, checkpoint_path: str = None) -> B2JointCheckpoint:
        return B2JointCheckpoint(
            input_path=checkpoint_path,
            epoch=0,
            player_model_state=self.player_model.state_dict(),
            img_model_state=self.img_head_model.state_dict(),
            player_optimizer_state=self.player_optimizer.state_dict(),
            img_optimizer_state=self.img_optimizer.state_dict(),
            player_scheduler_state=self.player_scheduler.state_dict(),
            img_scheduler_state=self.img_scheduler.state_dict(),
        )

    def _get_history(self, history_path: str = None) -> B2JointHistory:
        return B2JointHistory(history_path)

    def _train_mode(self):
        self.player_model.train()
        self.img_head_model.train()

    def _eval_mode(self):
        self.player_model.eval()
        self.img_head_model.eval()

    def _on_checkpoint_load(self) -> int:
        checkpoint: B2JointCheckpoint = self._checkpoint
        self.player_model.load_state_dict(checkpoint.player_model_state)
        self.img_head_model.load_state_dict(checkpoint.img_model_state)
        self.player_optimizer.load_state_dict(
            checkpoint.player_optimizer_state)
        self.img_optimizer.load_state_dict(checkpoint.img_optimizer_state)
        self.player_scheduler.load_state_dict(
            checkpoint.player_scheduler_state)
        self.img_scheduler.load_state_dict(checkpoint.img_scheduler_state)

    def _on_epoch_step(self, epoch: int):
        self.player_scheduler.step(self.player_val_loss)
        self.img_scheduler.step(self.img_val_loss)
        self._checkpoint.update_state(
            epoch=epoch,
            player_model_state=self.player_model.state_dict(),
            img_model_state=self.img_head_model.state_dict(),
            player_optimizer_state=self.player_optimizer.state_dict(),
            img_optimizer_state=self.img_optimizer.state_dict(),
            player_scheduler_state=self.player_scheduler.state_dict(),
            img_scheduler_state=self.img_scheduler.state_dict(),
        )
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

    def _save_trained_model(self):
        torch.save(
            {
                'player_model': self.player_model,
                'img_head_model': self.img_head_model
            },
            self._model_path
        )

    def _train_batch_step(self, inputs, labels):
        player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss = self.__get_outputs(
            inputs, labels)

        total_loss = player_loss + img_loss

        self.player_optimizer.zero_grad()
        self.img_optimizer.zero_grad()
        total_loss.backward()
        self.player_optimizer.step()
        self.img_optimizer.step()

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

        batch_size, frames_count, players_count, channels, width, height = inputs.shape
        inputs_view = inputs.view(
            batch_size*players_count, frames_count, channels, width, height
        )

        player_outputs, player_features = self.player_model(inputs_view)
        player_outputs = player_outputs.view(batch_size*players_count, -1)
        player_labels = player_labels.view(-1)
        player_loss = self.player_criterion(
            player_outputs, player_labels) / players_count

        player_features = player_features.view(batch_size, players_count, -1)
        img_outputs = self.img_head_model(player_features)
        img_loss = self.img_criterion(img_outputs, img_labels)

        return player_labels, player_outputs, player_loss, img_labels, img_outputs, img_loss

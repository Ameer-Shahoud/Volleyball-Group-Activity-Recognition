from typing import Type
import torch
from Abstracts.base_trainer import _BaseTrainer
from Baselines.B4_separate.b4_player_model import B4PlayerModel
from Enums.classification_level import ClassificationLevel
from Models.player_dataset import PlayerDataset


class B4PlayerTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix=ClassificationLevel.PLAYER.value,
            loss_labels=[ClassificationLevel.PLAYER.value]
        )

    def _get_dataset_type(self) -> Type[PlayerDataset]:
        return PlayerDataset

    def _get_model(self) -> B4PlayerModel:
        return B4PlayerModel()

    def _train_batch_step(self, inputs, labels):
        self._optimizers[0].zero_grad()
        _, outputs = self._model(inputs)
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        loss.backward()
        self._optimizers[0].step()

        self.train_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.train_correct[0] += (predicted == labels).sum().item()
        self.train_total[0] += labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        _, outputs = self._model(inputs)
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        self.val_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.val_correct[0] += (predicted == labels).sum().item()
        self.val_total[0] += labels.size(0)

    def _test_batch_step(self, inputs, labels):
        _, outputs = self._model(inputs)
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        self.test_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.test_correct[0] += (predicted == labels).sum().item()
        self.test_total[0] += labels.size(0)

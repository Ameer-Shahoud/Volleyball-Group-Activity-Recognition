from typing import Type
import torch
from Baselines.B2_separate.b2_player_model import B2PlayerModel
from Abstracts.base_trainer import _BaseTrainer
from Enums.classification_level import ClassificationLevel
from Models.player_dataset import PlayerDataset


class B2PlayerTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix=ClassificationLevel.PLAYER.value,
            loss_labels=[ClassificationLevel.PLAYER.value]
        )

    def _get_dataset_type(self) -> Type[PlayerDataset]:
        return PlayerDataset

    def _get_model(self) -> B2PlayerModel:
        return B2PlayerModel()

    def _train_batch_step(self, inputs, labels):
        self._optimizers[0].zero_grad()
        outputs = self.__map_outputs(self._model(inputs))
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        loss.backward()
        self._optimizers[0].step()

        self.train_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.train_correct[0] += (predicted == labels).sum().item()
        self.train_total[0] += labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        outputs = self.__map_outputs(self._model(inputs))
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        self.val_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.val_correct[0] += (predicted == labels).sum().item()
        self.val_total[0] += labels.size(0)

    def _test_batch_step(self, inputs, labels):
        outputs = self.__map_outputs(self._model(inputs))
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        self.test_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.test_correct[0] += (predicted == labels).sum().item()
        self.test_total[0] += labels.size(0)

    def __map_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, __ = outputs[0].shape
        return outputs[0].view(batch_size, -1)

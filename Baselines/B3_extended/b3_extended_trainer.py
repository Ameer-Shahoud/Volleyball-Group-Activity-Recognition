from typing import Type
import torch
from Abstracts.base_trainer import _BaseTrainer
from Baselines.B3_extended.b3_extended_model import B3ExtendedModel
from Enums.classification_level import ClassificationLevel
from Models.image_dataset import ImageDataset


class B3ExtendedTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            loss_labels=[ClassificationLevel.IMAGE.value]
        )

    def _get_dataset_type(self) -> Type[ImageDataset]:
        return ImageDataset

    def _get_model(self) -> B3ExtendedModel:
        return B3ExtendedModel()

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

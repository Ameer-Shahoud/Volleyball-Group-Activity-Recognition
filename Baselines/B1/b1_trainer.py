from typing import Type
import torch
from Baselines.B1.b1_history import B1History, B1HistoryItem
from Baselines.B1.b1_model import B1Model
from Models.base_trainer import _BaseTrainer
from Models.image_dataset import ImageDataset


class B1Trainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(checkpoint_path, history_path)

    def _init_values(self):
        self.train_loss, self.train_correct, self.train_total = 0.0, 0, 0
        self.val_loss, self.val_correct, self.val_total = 0.0, 0, 0
        self.test_loss, self.test_correct, self.test_total = 0.0, 0, 0

    def _get_dataset_type(self) -> Type[ImageDataset]:
        return ImageDataset

    def _get_model(self) -> B1Model:
        return B1Model()

    def _get_history(self, history_path: str = None) -> B1History:
        return B1History(history_path)

    def _on_epoch_step(self, epoch: int) -> B1HistoryItem:
        super()._on_epoch_step(epoch)
        self._schedulers[0].step(self.val_loss)
        return B1HistoryItem(
            epoch,
            self.train_loss / self.train_size,
            100 * self.train_correct / self.train_total,
            self.val_loss / self.val_size,
            100 * self.val_correct / self.val_total,
        )

    def _train_batch_step(self, inputs, labels):
        self._optimizers[0].zero_grad()
        outputs = self.__map_outputs(self._model(inputs))
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        loss.backward()
        self._optimizers[0].step()

        self.train_loss += loss.item()

        _, predicted = outputs.max(1)
        self.train_correct += (predicted == labels).sum().item()
        self.train_total += labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        outputs = self.__map_outputs(self._model(inputs))
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        self.val_loss += loss.item()

        _, predicted = outputs.max(1)
        self.val_correct += (predicted == labels).sum().item()
        self.val_total += labels.size(0)

    def _test_batch_step(self, inputs, labels):
        outputs = self.__map_outputs(self._model(inputs))
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        self.test_loss += loss.item()

        _, predicted = outputs.max(1)
        self.test_correct += (predicted == labels).sum().item()
        self.test_total += labels.size(0)

    def _on_test_step(self):
        print(
            f"Test Results:\nLoss: {self.test_loss/self.test_size:.4f}, Acc: {100 * self.test_correct/self.test_total:.2f}%"
        )

    def __map_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, __ = outputs[0].shape
        return outputs[0].view(batch_size, -1)

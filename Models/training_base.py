from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL.Image import Image
from tqdm import tqdm
from Models.config_mixin import _ConfigMixin
from Utils.cuda import get_device


class _TrainingBase(_ConfigMixin, ABC):
    def __init__(self):
        self._prepare_loaders()
        self._prepare_model()
        self._prepare_optimizer()
        self._reset_train_metrics()
        self._reset_val_metrics()
        self._reset_test_metrics()

    @abstractmethod
    def _prepare_loaders(self) -> None:
        pass

    @abstractmethod
    def _prepare_model(self) -> None:
        pass

    @abstractmethod
    def _prepare_optimizer(self) -> None:
        pass

    @abstractmethod
    def _train_mode(self) -> None:
        pass

    @abstractmethod
    def _eval_mode(self) -> None:
        pass

    @abstractmethod
    def _get_train_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def _get_val_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def _get_test_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def _train_step(self, inputs, labels) -> None:
        pass

    @abstractmethod
    def _eval_step(self, inputs, labels) -> None:
        pass

    @abstractmethod
    def _test_step(self, inputs, labels) -> None:
        pass

    @abstractmethod
    def _on_epoch_step(self) -> None:
        pass

    def _reset_train_metrics(self):
        self.loss, self.acc, self.correct, self.total = 0.0, 0.0, 0, 0

    def _reset_val_metrics(self):
        self.val_loss, self.val_acc,  self.val_correct, self.val_total = 0.0, 0.0, 0, 0

    def _reset_test_metrics(self):
        self.test_loss, self.test_acc,  self.test_correct, self.test_total = 0.0, 0.0, 0, 0

    def train(self):
        epochs = self.get_bl_cf().training.epochs
        for epoch in range(epochs):
            self._train_mode()
            self._reset_train_metrics()

            progress_bar = tqdm(self._get_train_loader(),
                                desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(
                    get_device()), labels.to(get_device())

                self._train_step(inputs, labels)

            self.acc = 100 * self.correct / self.total
            self.evaluate()

            print(f"\nTrain Loss: {self.loss/len(self._get_train_loader()):.4f} - Train Acc: {self.acc:.2f}% - Val Loss: {self.val_loss/len(self._get_val_loader()):.4f} - Val Acc: {self.val_acc:.2f}%\n")
            self._on_epoch_step()

    def evaluate(self):
        self._eval_mode()
        self._reset_val_metrics()

        with torch.no_grad():
            for inputs, labels in self._get_val_loader():
                inputs, labels = inputs.to(
                    get_device()), labels.to(get_device())
                self._eval_step(inputs, labels)

            self.val_acc = 100 * self.val_correct / self.val_total

    def test(self):
        self._eval_mode()
        self._reset_val_metrics()

        with torch.no_grad():
            for inputs, labels in self._get_test_loader():
                inputs, labels = inputs.to(
                    get_device()), labels.to(get_device())
                self._test_step(inputs, labels)

            self.test_acc = 100 * self.test_correct / self.test_total
            print(
                f"Test Results: Loss: {self.test_loss/len(self._get_test_loader()):.4f}, Acc: {self.test_acc:.2f}%"
            )

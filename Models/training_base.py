from abc import ABC, abstractmethod
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Models.config_mixin import _ConfigMixin
from Models.history import History
from Utils.cuda import get_device


class _TrainingBase(_ConfigMixin, ABC):
    """
    Abstract base class for training pipelines.
    It provides a consistent workflow for training, evaluation, and testing.
    """

    def __init__(self):
        """Initializes the training pipeline by preparing loaders, model, and optimizer."""
        self._prepare_loaders()
        self._prepare_model()
        self._prepare_optimizer()
        self._to_available_device()
        self._reset_train_metrics()
        self._reset_val_metrics()
        self._reset_test_metrics()
        self._checkpoint_path = os.path.join(
            self.get_bl_cf().output_dir, 'checkpoint.pth')
        self._model_path = os.path.join(
            self.get_bl_cf().output_dir, 'model.pth')

    @abstractmethod
    def _prepare_loaders(self) -> None:
        """
        Prepares DataLoaders for training, validation, and testing.

        - Loads ImageDataset for train, val, and test sets.
        - Initializes DataLoaders with the configured batch size."""
        pass

    @abstractmethod
    def _prepare_model(self) -> None:
        """
        Prepares model for training."""
        pass

    @abstractmethod
    def _prepare_optimizer(self) -> None:
        """
        Prepares optimizer and configure it."""
        pass

    @abstractmethod
    def _to_available_device(self) -> None:
        pass

    @abstractmethod
    def _get_train_loader(self) -> DataLoader:
        """
        Retrieves the DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training data.
        """
        pass

    @abstractmethod
    def _get_val_loader(self) -> DataLoader:
        """
        Retrieves the DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        pass

    @abstractmethod
    def _get_test_loader(self) -> DataLoader:
        """
        Retrieves the DataLoader for testing data.

        Returns:
            DataLoader: DataLoader for testing data.
        """
        pass

    @abstractmethod
    def _train_mode(self) -> None:
        """Sets baseline model to training mode."""
        pass

    @abstractmethod
    def _eval_mode(self) -> None:
        """Sets baseline model to evaluation mode."""
        pass

    @abstractmethod
    def _train_step(self, inputs, labels) -> None:
        """
        Defines a single training step.

        - Clears the gradients.
        - Performs a forward pass to get predictions.
        - Computes the loss.
        - Backpropagates the loss and updates model weights.
        - Calculates training accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        pass

    @abstractmethod
    def _eval_step(self, inputs, labels) -> None:
        """
        Defines a single evaluation step.

        - Performs a forward pass to get predictions.
        - Computes the loss.
        - Calculates validation accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        pass

    @abstractmethod
    def _test_step(self, inputs, labels) -> None:
        """
        Defines a single testing step.

        - Performs a forward pass to get predictions.
        - Computes the loss.
        - Calculates test accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        pass

    @abstractmethod
    def _load_last_checkpoint(self) -> int:
        pass

    @abstractmethod
    def _on_epoch_step(self, epoch: int) -> None:
        """A callback function emitted after each epoch."""
        pass

    @abstractmethod
    def _save_trained_model(self):
        pass

    def _reset_train_metrics(self):
        self.loss, self.acc, self.correct, self.total = 0.0, 0.0, 0, 0

    def _reset_val_metrics(self):
        self.val_loss, self.val_acc,  self.val_correct, self.val_total = 0.0, 0.0, 0, 0

    def _reset_test_metrics(self):
        self.test_loss, self.test_acc,  self.test_correct, self.test_total = 0.0, 0.0, 0, 0

    def _override(self):
        self.get_bl_cf().clear_output_dir()
        _epoch = 0
        _history = History()
        return _epoch, _history

    def train(self, override=False):
        self.clear_output()
        if override:
            _epoch, _history = self._override()
        else:
            try:
                _history = History().load()
                _epoch = self._load_last_checkpoint()
            except:
                print('Loading Checkpoint failed, Training started from begining')
                _epoch, _history = self._override()

        self.get_bl_cf().create_baseline_dir()
        epochs = self.get_bl_cf().training.epochs

        for epoch in range(_epoch, epochs):
            self._train_mode()
            self._reset_train_metrics()

            progress_bar = tqdm(self._get_train_loader(),
                                desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            self._to_available_device()
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(
                    get_device()), labels.to(get_device())

                self._train_step(inputs, labels)

            self.acc = 100 * self.correct / self.total
            self.evaluate()

            _history.add(
                self.loss,
                self.acc,
                self.val_loss,
                self.val_acc
            )
            _history.save()
            self._on_epoch_step(epoch)
            print(f"\nTrain Loss: {self.loss/len(self._get_train_loader()):.4f} - Train Acc: {self.acc:.2f}% - Val Loss: {self.val_loss/len(self._get_val_loader()):.4f} - Val Acc: {self.val_acc:.2f}%\n")
        self._save_trained_model()

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
        self._reset_test_metrics()

        with torch.no_grad():
            for inputs, labels in self._get_test_loader():
                inputs, labels = inputs.to(
                    get_device()), labels.to(get_device())
                self._test_step(inputs, labels)

            self.test_acc = 100 * self.test_correct / self.test_total
            print(
                f"Test Results: Loss: {self.test_loss/len(self._get_test_loader()):.4f}, Acc: {self.test_acc:.2f}%"
            )

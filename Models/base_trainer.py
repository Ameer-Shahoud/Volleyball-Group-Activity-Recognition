from abc import ABC, abstractmethod
import os
from typing import Type
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Enums.dataset_type import DatasetType
from Models.base_checkpoint import _BaseCheckpoint
from Models.base_dataset import _BaseDataset
from Models.config_mixin import _ConfigMixin
from Models.base_history import _BaseHistory, _BaseHistoryItem
from Utils.cuda import get_device


class _BaseTrainer(_ConfigMixin, ABC):
    """
    Abstract base class for training pipelines.
    It provides a consistent workflow for training, evaluation, and testing.
    """

    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None):
        """Initializes the training pipeline by preparing loaders, model, and optimizer."""
        self._init_values()
        self._prepare_loaders()
        self._prepare_model()
        self._prepare_optimizer()
        self._to_available_device()

        self._checkpoint = self._get_checkpoint(checkpoint_path)
        self._history = self._get_history(history_path)
        self._model_path = os.path.join(
            self.get_bl_cf().output_dir,
            f'model.{suffix}.pth' if suffix else 'model.pth'
        )

    @abstractmethod
    def _init_values(self) -> None:
        pass

    @abstractmethod
    def _get_dataset_type(self) -> Type[_BaseDataset]:
        pass

    def _prepare_loaders(self) -> None:
        train_dataset = self._get_dataset_type()(type=DatasetType.TRAIN)
        val_dataset = self._get_dataset_type()(type=DatasetType.VAL)
        test_dataset = self._get_dataset_type()(type=DatasetType.TEST)

        self.train_size = len(train_dataset)
        self.val_size = len(val_dataset)
        self.test_size = len(test_dataset)

        batch_size = self.get_bl_cf().training.batch_size

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

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
    def _get_checkpoint(self, checkpoint_path: str = None) -> _BaseCheckpoint:
        pass

    @abstractmethod
    def _get_history(self, history_path: str = None) -> _BaseHistory:
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
    def _train_batch_step(self, inputs, labels) -> None:
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
    def _eval_batch_step(self, inputs, labels) -> None:
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
    def _on_epoch_step(self, epoch: int) -> _BaseHistoryItem:
        """A callback function emitted after each epoch."""
        pass

    @abstractmethod
    def _test_batch_step(self, inputs, labels) -> None:
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
    def _on_test_step(self) -> None:
        pass

    @abstractmethod
    def _on_checkpoint_load(self) -> int:
        pass

    @abstractmethod
    def _save_trained_model(self):
        pass

    def train(self, override=False):
        self.get_bl_cf().create_baseline_dir()
        self.clear_output()

        if override:
            self._history.reset()
            self._checkpoint.reset()
        else:
            try:
                self._history.load(from_input=True)
                self._checkpoint.load(from_input=True)
                self._on_checkpoint_load()
            except:
                print('Loading Checkpoint failed, Training started from begining')
                self._history.reset()
                self._checkpoint.reset()

        self._to_available_device()

        epochs = self.get_bl_cf().training.epochs
        for epoch in range(self._checkpoint.epoch, epochs):
            self._train_mode()

            progress_bar = tqdm(self.train_loader,
                                desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = self._inputs_to_device(inputs, labels)

                self._train_batch_step(inputs, labels)

            self.evaluate()
            history_item = self._on_epoch_step(epoch)
            self._history.add(history_item)
            self._checkpoint.save()
            self._init_values()

        self._save_trained_model()

    def evaluate(self):
        self._eval_mode()

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = self._inputs_to_device(inputs, labels)
                self._eval_batch_step(inputs, labels)

    def test(self):
        self._eval_mode()
        self._init_values()
        self._to_available_device()

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = self._inputs_to_device(inputs, labels)
                self._test_batch_step(inputs, labels)
            self._on_test_step()

    def _inputs_to_device(self, inputs, labels):
        if isinstance(inputs, (tuple, list)):
            inputs = tuple([_input.to(get_device())
                            for _input in inputs])
        else:
            inputs = inputs.to(get_device())
        if isinstance(labels, (tuple, list)):
            labels = tuple([_label.to(get_device())
                            for _label in labels])
        else:
            labels = labels.to(get_device())

        return inputs, labels

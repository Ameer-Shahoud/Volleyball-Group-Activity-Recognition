from abc import ABC, abstractmethod
import os
from typing import Type
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Enums.dataset_type import DatasetType
from Abstracts.base_checkpoint import _BaseCheckpoint
from Abstracts.base_dataset import _BaseDataset
from Abstracts.base_model import _BaseModel
from Models.checkpoint import Checkpoint
from Abstracts.config_mixin import _ConfigMixin
from Abstracts.base_history import _BaseHistory, _BaseHistoryItem
from Models.history import History, HistoryItem
from Utils.cuda import get_device


class _BaseTrainer(_ConfigMixin, ABC):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None, loss_labels: list[str] = []):
        self._suffix = suffix
        self._loss_labels = loss_labels

        self._prepare_loaders()
        self._init_values()

        self._model = self._get_model()
        self._criterions = self.get_criterions()
        self._optimizers = self.get_optimizers()
        self._schedulers = self.get_schedulers()
        self._checkpoint = self._get_checkpoint(checkpoint_path)
        self._history = self._get_history(history_path)
        self._model_path = os.path.join(
            self.get_bl_cf().output_dir,
            f'model.{suffix}.pth' if suffix else 'model.pth'
        )

        self._to_available_device()

    def _init_values(self) -> None:
        l = len(self._loss_labels)
        self.train_loss, self.train_correct, self.train_total = [
            0.0] * l, [0] * l, [0] * l
        self.val_loss, self.val_correct, self.val_total = [
            0.0] * l, [0] * l, [0] * l
        self.test_loss, self.test_correct, self.test_total = [
            0.0] * l, [0] * l, [0] * l

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
    def _get_model(self) -> _BaseModel:
        pass

    def get_criterions(self) -> list[nn.Module]:
        return [nn.CrossEntropyLoss()]

    def get_optimizers(self) -> list[torch.optim.Optimizer]:
        return [torch.optim.Adam(
            (p for p in self._model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )]

    def get_schedulers(self) -> list[torch.optim.lr_scheduler.LRScheduler]:
        return [torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizers[0], mode='min', factor=0.1, patience=2
        )]

    def _get_checkpoint(self, checkpoint_path: str = None) -> _BaseCheckpoint:
        return Checkpoint(
            input_path=checkpoint_path,
            suffix=self._suffix,
            epoch=0,
            model_state=self._model.state_dict(),
            criterions_state=list(
                map(lambda c: c.state_dict(), self._criterions)
            ),
            optimizers_state=list(
                map(lambda o: o.state_dict(), self._optimizers)
            ),
            schedulers_state=list(
                map(lambda s: s.state_dict(), self._schedulers)
            ),
        )

    def _get_history(self, history_path: str = None) -> _BaseHistory:
        return History(history_path, self._suffix)

    def _to_available_device(self) -> None:
        self._model.to(get_device())
        [criterion.to(get_device()) for criterion in self._criterions]
        for optimizer in self._optimizers:
            for state in optimizer.state.values():
                if isinstance(state, torch.Tensor):
                    state.data = state.data.to(get_device())

    def _on_checkpoint_load(self) -> None:
        checkpoint: Checkpoint = self._checkpoint
        self._model.load_state_dict(checkpoint.model_state)
        [self._criterions[i].load_state_dict(
            checkpoint.criterions_state[i]) for i in range(len(self._criterions))]
        [self._optimizers[i].load_state_dict(
            checkpoint.optimizers_state[i]) for i in range(len(self._optimizers))]
        [self._schedulers[i].load_state_dict(
            checkpoint.schedulers_state[i]) for i in range(len(self._schedulers))]

    def _train_mode(self) -> None:
        self._model.train()
        [criterion.train() for criterion in self._criterions]

    def _eval_mode(self) -> None:
        self._model.eval()
        [criterion.eval() for criterion in self._criterions]

    @abstractmethod
    def _train_batch_step(self, inputs, labels) -> None:
        pass

    @abstractmethod
    def _eval_batch_step(self, inputs, labels) -> None:
        pass

    @abstractmethod
    def _test_batch_step(self, inputs, labels) -> None:
        pass

    def _on_epoch_step(self, epoch: int) -> _BaseHistoryItem:
        self._checkpoint.update_state(
            epoch=epoch,
            model_state=self._model.state_dict(),
            criterions_state=list(
                map(lambda c: c.state_dict(), self._criterions)
            ),
            optimizers_state=list(
                map(lambda o: o.state_dict(), self._optimizers)
            ),
            schedulers_state=list(
                map(lambda s: s.state_dict(), self._schedulers)
            ),
        )

        if len(self._schedulers):
            for i in range(len(self._schedulers)):
                self._schedulers[i].step(self.val_loss[i])

        r = range(len(self._loss_labels))
        return HistoryItem(
            epoch,
            [self.train_loss[i] / self.train_size for i in r],
            [100 * self.train_correct[i] / self.train_total[i] for i in r],
            [self.val_loss[i] / self.val_size for i in r],
            [100 * self.val_correct[i] / self.val_total[i] for i in r],
            labels=self._loss_labels
        )

    def _on_test_step(self) -> None:
        print(
            "Test Results:\n",
            *[f"{self._loss_labels[i].capitalize()}  ---   Loss: {self.test_loss[i]/self.test_size:.4f}, Acc: {100 * self.test_correct[i]/self.test_total[i]:.2f}%" for i in range(len(self._loss_labels))]
        )

    def _save_trained_model(self) -> None:
        torch.save(self._model, self._model_path)

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

    def plot_history(self):
        self._history.plot_history()

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

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
from Abstracts.base_history import _BaseHistory
from Models.early_stopping import EarlyStopping
from Models.history import History, HistoryItem
from Models.metrics import Metrics
from Models.test_results import TestResults
from Utils.cuda import get_device


class _BaseTrainer(_ConfigMixin, ABC):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None, loss_levels: list[str] = []):
        self._suffix = suffix
        self._loss_levels = loss_levels
        self._early_stopping = EarlyStopping()

        self._prepare_loaders_and_metrics()

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

        self._init_trainer()
        self._to_available_device()

    def _prepare_loaders_and_metrics(self) -> None:
        train_dataset = self._get_dataset_type()(type=DatasetType.TRAIN)
        val_dataset = self._get_dataset_type()(type=DatasetType.VAL)
        test_dataset = self._get_dataset_type()(type=DatasetType.TEST)

        self.train_metrics = Metrics(self._loss_levels, len(train_dataset))
        self.val_metrics = Metrics(self._loss_levels, len(val_dataset))
        self.test_metrics = Metrics(self._loss_levels, len(test_dataset))

        batch_size = self.get_bl_cf().training.batch_size

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

    @abstractmethod
    def _get_dataset_type(self) -> Type[_BaseDataset]:
        pass

    @abstractmethod
    def _get_model(self) -> _BaseModel:
        pass

    @abstractmethod
    def get_criterions(self) -> list[nn.Module]:
        pass

    @abstractmethod
    def get_optimizers(self) -> list[torch.optim.Optimizer]:
        pass

    @abstractmethod
    def get_schedulers(self) -> list[torch.optim.lr_scheduler.LRScheduler]:
        pass

    def _get_checkpoint(self, checkpoint_path: str = None) -> _BaseCheckpoint:
        return Checkpoint(
            input_path=checkpoint_path,
            suffix=self._suffix,
            epoch=0,
            model_state=self._model.state_dict(),
            best_model_state=self._model.state_dict(),
            best_model_metrics=None,
            early_stopping_state=None,
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

    def _init_trainer(self) -> None:
        try:
            self._history.load(from_input=True)
            self._checkpoint.load(from_input=True)
            self._checkpoint.epoch += 1
            self._on_checkpoint_load()
            print('Checkpoint and history loaded successfully.')
        except:
            self._history.reset()
            self._checkpoint.reset()
            print('Checkpoint and history loading failed.')

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

        if checkpoint.early_stopping_state:
            self._early_stopping = checkpoint.early_stopping_state

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
    def _batch_step(self, metrics: Metrics, inputs: torch.Tensor, labels: torch.Tensor, apply_backward=False) -> None:
        pass

    def train(self, override=False):
        self.get_bl_cf().create_baseline_dir()
        self.clear_output()

        if override:
            self._history.reset()
            self._checkpoint.reset()
        else:
            self._init_trainer()

        self._to_available_device()
        epochs = self.get_bl_cf().training.epochs

        for epoch in range(self._checkpoint.epoch, epochs):
            if self._early_stopping.early_stop:
                self.get_bl_cf().logger.warning(
                    f"Early stopping triggered at epoch {epoch}!"
                )
                break
            self.train_metrics.reset()
            self.val_metrics.reset()
            self._train_mode()

            progress_bar = tqdm(self.train_loader,
                                desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = self._inputs_to_device(inputs, labels)

                self._batch_step(
                    self.train_metrics, inputs, labels, apply_backward=True
                )

            self.evaluate()
            self._early_stopping(
                self.val_metrics.get_early_stopping_metric(
                    self._loss_levels[-1], self._early_stopping.metric
                )
            )
            self._on_train_epoch_step(epoch)

        self.get_bl_cf().logger.log_to_tensorboard()
        self._save_trained_model()

    def evaluate(self):
        self._eval_mode()

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = self._inputs_to_device(inputs, labels)
                self._batch_step(self.val_metrics, inputs, labels)

    def _on_train_epoch_step(self, epoch: int) -> None:
        self._checkpoint.update_state(
            epoch=epoch,
            model_state=self._model.state_dict(),
            best_model_state=self._model.state_dict(
            ) if self._early_stopping.improved else self._checkpoint.best_model_state,
            best_model_metrics=self.val_metrics if self._early_stopping.improved else self._checkpoint.best_model_metrics,
            early_stopping_state=self._early_stopping,
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
        self._checkpoint.save()

        if len(self._schedulers):
            for i in range(len(self._schedulers)):
                self._schedulers[i].step(
                    self.val_metrics.get_loss(self._loss_levels[i])
                )

        l = range(len(self._loss_levels))
        self._history.add(
            HistoryItem(
                epoch+1,
                self.train_metrics.copy(),
                self.val_metrics.copy(),
                levels=self._loss_levels
            )
        )

    def test(self):
        self._eval_mode()
        self.test_metrics.reset()
        self._to_available_device()

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = self._inputs_to_device(inputs, labels)
                self._batch_step(self.test_metrics, inputs, labels)
            self._on_test_full_step()

    def _on_test_full_step(self) -> None:
        l = range(len(self._loss_levels))
        test_results = TestResults(
            test_metrics=self.test_metrics.copy(),
            suffix=self._suffix,
            levels=self._loss_levels
        )
        self.get_bl_cf().logger.log_to_tensorboard()
        test_results.plot_confustion_matrix()
        test_results.print_classification_report()
        test_results.save()

    def _save_trained_model(self) -> None:
        torch.save(self._model, self._model_path)

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

    def plot_history(self):
        self._history.plot_history()

    def save_version(self, checkpoint_path: str = None, history_path: str = None, best_model=False) -> None:
        self.get_bl_cf().create_baseline_dir()

        history = self._get_history(history_path).load(from_input=True)
        checkpoint = self._get_checkpoint(
            checkpoint_path
        ).load(from_input=True)
        model = self._get_model()
        model.load_state_dict(
            checkpoint.best_model_state if best_model else checkpoint.model_state
        )

        history.save()
        checkpoint.save()
        torch.save(model, self._model_path)
        print("Version saved successfully.")

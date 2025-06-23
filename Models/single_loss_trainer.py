import torch
from Abstracts.base_trainer import _BaseTrainer
from Models.metrics import Metrics
from torch import nn


class SingleLossTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None, loss_levels: list[str] = []):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix,
            loss_levels
        )

    def get_criterions(self) -> list[nn.Module]:
        return [nn.CrossEntropyLoss()]

    def get_optimizers(self) -> list[torch.optim.Optimizer]:
        return [torch.optim.Adam(
            (p for p in self._model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )]

    def get_schedulers(self) -> list[torch.optim.lr_scheduler.LRScheduler]:
        config = self.get_bl_cf().training.scheduler
        return [torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizers[0], mode=config.mode, factor=config.factor, patience=config.patience
        )]

    def _batch_step(self, metrics: Metrics, inputs: torch.Tensor, labels: torch.Tensor, apply_backward=False):
        outputs: torch.Tensor = self._model(inputs)
        batch_size = outputs.shape[0]
        outputs = outputs.view(batch_size, -1)
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        if apply_backward:
            loss.backward()
            self._optimizers[0].step()

        loss = loss.item()
        _, predicted = outputs.max(1)

        metrics.update_metrics(
            level=self._loss_levels[0],
            loss=loss,
            predicted=predicted,
            labels=labels
        )

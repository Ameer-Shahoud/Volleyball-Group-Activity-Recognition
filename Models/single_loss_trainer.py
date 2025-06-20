import torch
from Abstracts.base_trainer import _BaseTrainer


class SingleLossTrainer(_BaseTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None, suffix: str = None, loss_labels: list[str] = []):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix,
            loss_labels
        )

    def _train_batch_step(self, inputs, labels):
        loss, correct, total = self._batch_step(
            inputs, labels, apply_backward=True
        )

        self.train_loss[0] += loss
        self.train_correct[0] += correct
        self.train_total[0] += total

    def _eval_batch_step(self, inputs, labels):
        loss, correct, total = self._batch_step(inputs, labels)

        self.val_loss[0] += loss
        self.val_correct[0] += correct
        self.val_total[0] += total

    def _test_batch_step(self, inputs, labels):
        loss, correct, total = self._batch_step(inputs, labels)

        self.test_loss[0] += loss
        self.test_correct[0] += correct
        self.test_total[0] += total

    def _batch_step(self, inputs: torch.Tensor, labels: torch.Tensor, apply_backward=False) -> tuple:
        outputs: torch.Tensor = self._model(inputs)
        batch_size = outputs.shape[0]
        outputs = outputs.view(batch_size, -1)
        loss: torch.Tensor = self._criterions[0](outputs, labels)

        if apply_backward:
            loss.backward()
            self._optimizers[0].step()

        loss = loss.item()

        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)

        return loss, correct, total

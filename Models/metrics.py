from sklearn.metrics import classification_report
import torch
from torchmetrics import ConfusionMatrix, F1Score
from Abstracts.config_mixin import _ConfigMixin
from Types.metric_type import MetricType
from Utils.cuda import get_device


class Metrics(_ConfigMixin):
    def __init__(self, loss_levels: list[str], size: int):
        self._loss_levels = loss_levels
        self._size = size

        l = len(loss_levels)
        num_classes = [
            len(self.get_cf().dataset.get_categories(level)) for level in self._loss_levels
        ]

        self.__loss = [0.0] * l
        self.__correct = [0] * l
        self.__total = [0] * l
        self.__f1 = [
            F1Score(task='multiclass', num_classes=num, average='weighted').to(get_device()).eval() for num in num_classes
        ]

        self.__labels = [torch.Tensor([]).to(get_device())] * l
        self.__predicted = [torch.Tensor([]).to(get_device())] * l

    def __get_idx(self, level: str):
        return self._loss_levels.index(level)

    def update_metrics(
        self,
        level: str,
        loss: float,
        predicted: torch.Tensor,
        labels: torch.Tensor
    ):
        idx = self.__get_idx(level)
        self.__loss[idx] += loss

        self.__correct[idx] += (predicted == labels).sum().item()
        self.__total[idx] += labels.size(0)

        self.__f1[idx].update(predicted, labels)

        self.__labels[idx] = torch.cat((self.__labels[idx], labels))
        self.__predicted[idx] = torch.cat((self.__predicted[idx], predicted))

    def reset(self):
        self.__init__(self._loss_levels, self._size)

    def get_loss(self, level: str):
        idx = self.__get_idx(level)
        return self.__loss[idx] / self._size

    def get_acc(self, level: str):
        idx = self.__get_idx(level)
        return (100 * self.__correct[idx] / self.__total[idx]) if self.__total[idx] else 0

    def get_f1(self, level: str):
        idx = self.__get_idx(level)
        return self.__f1[idx].compute().cpu()

    def get_confusion_matrix(self, level: str, normalized=True):
        idx = self.__get_idx(level)
        num_classes = len(self.get_cf().dataset.get_categories(level))

        cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)(
            self.__predicted[idx], self.__labels[idx]
        )

        if normalized:
            cm = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1e-6)

        return cm

    def get_classification_report(self, level: str):
        idx = self.__get_idx(level)
        return classification_report(
            y_true=self.__labels[idx].cpu().numpy(),
            y_pred=self.__predicted[idx].cpu().numpy(),
            labels=self.get_cf().dataset.get_categories(level),
            target_names=self.get_cf().dataset.get_categories(level),
        )

    def get_early_stopping_metric(self, level: str, metric: MetricType):
        return {
            'loss': self.get_loss(level),
            'acc': self.get_acc(level),
            'f1': self.get_f1(level),
        }[metric]

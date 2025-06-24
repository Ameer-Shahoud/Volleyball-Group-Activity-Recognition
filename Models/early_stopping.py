from typing import Any
import numpy as np
from Abstracts.config_mixin import _ConfigMixin
from Types.metric_type import MetricType
from Types.mode_type import ModeType


class EarlyStopping(_ConfigMixin):
    def __init__(
        self,
        metric: MetricType = None,
        patience: int = None,
        delta: float = None,
        mode: ModeType = None
    ):
        self.metric = metric if metric else self.get_bl_cf().training.early_stopping.metric
        self.patience = patience if patience else self.get_bl_cf(
        ).training.early_stopping.patience
        self.delta = delta if delta else self.get_bl_cf().training.early_stopping.delta
        self.mode = mode if mode else self.get_bl_cf().training.early_stopping.mode

        self.counter = 0
        self.best_metric = np.Inf if mode == 'min' else -np.Inf
        self.early_stop = False
        self.improved = False

    def __call__(self, current_metric:  Any):
        self.improved = False
        if self.mode == 'min':
            improved = current_metric < (
                self.best_metric - self.delta)
        else:
            improved = current_metric > (
                self.best_metric + self.delta)

        if improved:
            self.improved = True
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def __setstate__(self, state):
        self.__dict__.update(state)
        _patience = self.get_bl_cf().training.early_stopping.patience
        if _patience and _patience > self.patience:
            self.early_stop = False
        self.patience = _patience if _patience else self.patience

        _delta = self.get_bl_cf().training.early_stopping.delta
        self.delta = _delta if _delta else self.delta

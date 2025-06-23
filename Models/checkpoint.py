from typing import Any
from Abstracts.base_checkpoint import _BaseCheckpoint
from Models.early_stopping import EarlyStopping
from Models.metrics import Metrics
from Types.metric_type import MetricType


class Checkpoint(_BaseCheckpoint):
    def __init__(
        self,
        input_path: str = None,
        suffix: str = None,
        epoch=0,
        model_state: dict[str, Any] = {},
        best_model_state: dict[str, Any] = {},
        best_model_metrics: Metrics = None,
        early_stopping_state: EarlyStopping = None,
        criterions_state: list[dict[str, Any]] = {},
        optimizers_state: list[dict[str, Any]] = {},
        schedulers_state: list[dict[str, Any]] = {},
    ):
        super().__init__(
            input_path=input_path,
            suffix=suffix,
            epoch=epoch,
            model_state=model_state,
            best_model_state=best_model_state,
            best_model_metrics=best_model_metrics,
            early_stopping_state=early_stopping_state,
            criterions_state=criterions_state,
            optimizers_state=optimizers_state,
            schedulers_state=schedulers_state,
        )

    def update_state(
        self,
        epoch=0,
        model_state: dict[str, Any] = {},
        best_model_state: dict[str, Any] = {},
        best_model_metrics: Metrics = None,
        early_stopping_state: EarlyStopping = None,
        criterions_state: list[dict[str, Any]] = {},
        optimizers_state: list[dict[str, Any]] = {},
        schedulers_state: list[dict[str, Any]] = {},
    ):
        self.epoch = epoch
        self.model_state = model_state
        self.best_model_state = best_model_state
        self.best_model_metrics = best_model_metrics
        self.early_stopping_state = early_stopping_state
        self.criterions_state = criterions_state
        self.optimizers_state = optimizers_state
        self.schedulers_state = schedulers_state

    def _get_state_dict(self):
        return {
            'epoch': self.epoch,
            'model_state': self.model_state,
            'best_model_state': self.best_model_state,
            'best_model_metrics': self.best_model_metrics,
            'early_stopping_state': self.early_stopping_state,
            'criterions_state': self.criterions_state,
            'optimizers_state': self.optimizers_state,
            'schedulers_state': self.schedulers_state,
        }

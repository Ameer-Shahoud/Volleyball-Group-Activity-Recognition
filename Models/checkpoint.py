from typing import Any
from Models.base_checkpoint import _BaseCheckpoint


class Checkpoint(_BaseCheckpoint):
    def __init__(
        self,
        input_path: str = None,
        epoch=0,
        model_state: dict[str, Any] = {},
        criterions_state: list[dict[str, Any]] = {},
        optimizers_state: list[dict[str, Any]] = {},
        schedulers_state: list[dict[str, Any]] = {}
    ):
        super().__init__(
            input_path=input_path,
            epoch=epoch,
            model_state=model_state,
            criterions_state=criterions_state,
            optimizers_state=optimizers_state,
            schedulers_state=schedulers_state
        )

    def update_state(
        self,
        epoch=0,
        model_state: dict[str, Any] = {},
        criterions_state: list[dict[str, Any]] = {},
        optimizers_state: list[dict[str, Any]] = {},
        schedulers_state: list[dict[str, Any]] = {}
    ):
        self.epoch = epoch
        self.model_state = model_state
        self.criterions_state = criterions_state
        self.optimizers_state = optimizers_state
        self.schedulers_state = schedulers_state

    def _get_state_dict(self):
        return {
            'epoch': self.epoch,
            'model_state': self.model_state,
            'criterions_state': self.criterions_state,
            'optimizers_state': self.optimizers_state,
            'schedulers_state': self.schedulers_state,
        }

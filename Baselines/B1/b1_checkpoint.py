from typing import Any
from Models.base_checkpoint import _BaseCheckpoint


class B1Checkpoint(_BaseCheckpoint):
    def __init__(
        self,
        input_path: str = None,
        epoch=0,
        model_state: dict[str, Any] = {},
        optimizer_state: dict[str, Any] = {},
        scheduler_state: dict[str, Any] = {}
    ):
        super().__init__(
            input_path=input_path,
            epoch=epoch,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state
        )

    def update_state(
        self,
        epoch=0,
        model_state: dict[str, Any] = {},
        optimizer_state: dict[str, Any] = {},
        scheduler_state: dict[str, Any] = {}
    ):
        self.epoch = epoch
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state

    def _get_state_dict(self):
        return {
            'epoch': self.epoch,
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'scheduler_state': self.scheduler_state,
        }

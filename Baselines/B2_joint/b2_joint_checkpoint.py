from typing import Any
from Models.base_checkpoint import _BaseCheckpoint


class B2JointCheckpoint(_BaseCheckpoint):
    def __init__(
        self,
        input_path: str = None,
        epoch=0,
        player_model_state: dict[str, Any] = {},
        img_model_state: dict[str, Any] = {},
        player_optimizer_state: dict[str, Any] = {},
        img_optimizer_state: dict[str, Any] = {},
        player_scheduler_state: dict[str, Any] = {},
        img_scheduler_state: dict[str, Any] = {}
    ):
        super().__init__(
            input_path=input_path,
            epoch=epoch,
            player_model_state=player_model_state,
            img_model_state=img_model_state,
            player_optimizer_state=player_optimizer_state,
            img_optimizer_state=img_optimizer_state,
            player_scheduler_state=player_scheduler_state,
            img_scheduler_state=img_scheduler_state,
        )

    def update_state(
        self,
        epoch=0,
        player_model_state: dict[str, Any] = {},
        img_model_state: dict[str, Any] = {},
        player_optimizer_state: dict[str, Any] = {},
        img_optimizer_state: dict[str, Any] = {},
        player_scheduler_state: dict[str, Any] = {},
        img_scheduler_state: dict[str, Any] = {}
    ):
        self.epoch = epoch
        self.player_model_state = player_model_state
        self.img_model_state = img_model_state
        self.player_optimizer_state = player_optimizer_state
        self.img_optimizer_state = img_optimizer_state
        self.player_scheduler_state = player_scheduler_state
        self.img_scheduler_state = img_scheduler_state

    def _get_state_dict(self):
        return {
            'epoch': self.epoch,
            'player_model_state': self.player_model_state,
            'img_model_state': self.img_model_state,
            'player_optimizer_state': self.player_optimizer_state,
            'img_optimizer_state': self.img_optimizer_state,
            'player_scheduler_state': self.player_scheduler_state,
            'img_scheduler_state': self.img_scheduler_state,
        }

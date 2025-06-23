from typing import Type
from Baselines.B3_separate.Player.b3_player_model import B3PlayerModel
from Enums.classification_level import ClassificationLevel
from Models.player_dataset import PlayerDataset
from Models.single_loss_trainer import SingleLossTrainer


class B3PlayerTrainer(SingleLossTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            suffix=ClassificationLevel.PLAYER.value,
            loss_levels=[ClassificationLevel.PLAYER.value]
        )

    def _get_dataset_type(self) -> Type[PlayerDataset]:
        return PlayerDataset

    def _get_model(self) -> B3PlayerModel:
        return B3PlayerModel()

from typing import Type
from Baselines.B6_Separate.Image.b6_img_model import B6ImgModel
from Enums.classification_level import ClassificationLevel
from Models.image_players_dataset import ImagePlayersDataset
from Models.single_loss_trainer import SingleLossTrainer


class B6ImgTrainer(SingleLossTrainer):
    def __init__(self, pretrained_player_model_path: str, checkpoint_path: str = None, history_path: str = None):
        self.pretrained_player_model_path = pretrained_player_model_path

        super().__init__(
            checkpoint_path,
            history_path,
            suffix=ClassificationLevel.IMAGE.value,
            loss_levels=[ClassificationLevel.IMAGE.value]
        )

    def _get_dataset_type(self) -> Type[ImagePlayersDataset]:
        return ImagePlayersDataset

    def _get_model(self) -> B6ImgModel:
        return B6ImgModel(self.pretrained_player_model_path)

from typing import Type
from Baselines.B4.b4_model import B4Model
from Enums.classification_level import ClassificationLevel
from Models.image_dataset import ImageDataset
from Models.single_loss_trainer import SingleLossTrainer


class B4Trainer(SingleLossTrainer):
    def __init__(self, pretrained_img_model_path: str, checkpoint_path: str = None, history_path: str = None):
        self.pretrained_img_model_path = pretrained_img_model_path

        super().__init__(
            checkpoint_path,
            history_path,
            loss_levels=[ClassificationLevel.IMAGE.value]
        )

    def _get_dataset_type(self) -> Type[ImageDataset]:
        return ImageDataset

    def _get_model(self) -> B4Model:
        return B4Model(self.pretrained_img_model_path)

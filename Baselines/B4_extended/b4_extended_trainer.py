from typing import Type
from Baselines.B4_extended.b4_extended_model import B4ExtendedModel
from Enums.classification_level import ClassificationLevel
from Models.image_dataset import ImageDataset
from Models.single_loss_trainer import SingleLossTrainer


class B4ExtendedTrainer(SingleLossTrainer):
    def __init__(self, checkpoint_path: str = None, history_path: str = None):
        super().__init__(
            checkpoint_path,
            history_path,
            loss_labels=[ClassificationLevel.IMAGE.value]
        )

    def _get_dataset_type(self) -> Type[ImageDataset]:
        return ImageDataset

    def _get_model(self) -> B4ExtendedModel:
        return B4ExtendedModel()

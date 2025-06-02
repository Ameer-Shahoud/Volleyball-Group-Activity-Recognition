from typing import Type
from Abstracts.base_trainer import _BaseTrainer
from Baselines.B4_separate.b4_img_model import B4ImgModel
from Enums.classification_level import ClassificationLevel
from Models.image_players_dataset import ImagePlayersDataset


class B4ImgTrainer(_BaseTrainer):
    def __init__(self, pretrained_player_model_path: str, checkpoint_path: str = None, history_path: str = None):
        self.pretrained_player_model_path = pretrained_player_model_path

        super().__init__(
            checkpoint_path,
            history_path,
            suffix=ClassificationLevel.IMAGE.value,
            loss_labels=[ClassificationLevel.IMAGE.value]
        )

    def _get_dataset_type(self) -> Type[ImagePlayersDataset]:
        return ImagePlayersDataset

    def _get_model(self) -> B4ImgModel:
        return B4ImgModel(self.pretrained_player_model_path)

    def _train_batch_step(self, inputs, labels):
        self._optimizers[0].zero_grad()
        labels, outputs, loss = self.__get_outputs(inputs, labels)

        loss.backward()
        self._optimizers[0].step()

        self.train_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.train_correct[0] += (predicted == labels).sum().item()
        self.train_total[0] += labels.size(0)

    def _eval_batch_step(self, inputs, labels):
        labels, outputs, loss = self.__get_outputs(inputs, labels)

        self.val_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.val_correct[0] += (predicted == labels).sum().item()
        self.val_total[0] += labels.size(0)

    def _test_batch_step(self, inputs, labels):
        labels, outputs, loss = self.__get_outputs(inputs, labels)

        self.test_loss[0] += loss.item()

        _, predicted = outputs.max(1)
        self.test_correct[0] += (predicted == labels).sum().item()
        self.test_total[0] += labels.size(0)

    def __get_outputs(self, inputs, labels):
        _, img_labels = labels
        img_outputs = self._model(inputs)
        img_loss = self._criterions[0](img_outputs, img_labels)

        return img_labels, img_outputs, img_loss

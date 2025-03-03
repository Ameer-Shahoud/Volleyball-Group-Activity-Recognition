import os
import pickle

from Models.config_mixin import _ConfigMixin


class History(_ConfigMixin):
    def __init__(
        self,
        train_loss: list[float] = [],
        train_acc: list[float] = [],
        val_loss: list[float] = [],
        val_acc: list[float] = [],
        filename='history.pkl'
    ):
        if len(set([len(train_loss), len(train_acc), len(val_loss), len(val_acc),])) != 1:
            raise ValueError(
                'lists should be initialized with same history steps length')

        self.__train_loss: list[float] = train_loss
        self.__train_acc: list[float] = train_acc
        self.__val_loss: list[float] = val_loss
        self.__val_acc: list[float] = val_acc

        self.__path = os.path.join(self.get_bl_cf().output_dir, filename)

    def add(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float
    ):
        self.__train_loss.append(train_loss)
        self.__train_acc.append(train_acc)
        self.__val_loss.append(val_loss)
        self.__val_acc.append(val_acc)

    def get_history(self) -> dict[str, list[float]]:
        return {
            'train_loss': self.__train_loss,
            'train_acc': self.__train_acc,
            'val_loss': self.__val_loss,
            'val_acc': self.__val_acc,
        }

    def save(self):
        with open(self.__path, "wb") as f:
            pickle.dump(self, f)

    def load(self) -> 'History':
        try:
            with open(self.__path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No previous '{self.__path}' history found, starting fresh.")

from abc import ABC, abstractmethod
import os
import pickle
from Abstracts.config_mixin import _ConfigMixin


class _BaseHistory(_ConfigMixin, ABC):
    def __init__(
        self,
        input_path: str = None,
        suffix: str = None
    ):
        super().__init__()

        self._input_path = input_path if input_path else os.path.join(
            self.get_bl_cf().output_dir,
            f'history.{suffix}.pkl' if suffix else 'history.pkl'
        )

        self._output_path = os.path.join(
            self.get_bl_cf().output_dir,
            f'history.{suffix}.pkl' if suffix else 'history.pkl'
        )

        self._fig_output_path = os.path.join(
            self.get_bl_cf().output_dir,
            f'history-figure.{suffix}.png' if suffix else 'history-figure.png'
        )

        self._history_list: list[_BaseHistoryItem] = []

    def add(
        self,
        item: '_BaseHistoryItem',
        update_file=True,
        print_item=True,
    ):
        self._history_list.append(item)
        if update_file:
            self.save()
        if print_item:
            self.get_bl_cf().logger.info(item)

    def list(self) -> list['_BaseHistoryItem']:
        return self._history_list

    def reset(self, remove_file=True) -> None:
        self._history_list = []
        if remove_file and os.path.exists(self._output_path):
            os.remove(self._output_path)

    def save(self):
        with open(self._output_path, "wb") as f:
            pickle.dump(self, f)

    def load(self, from_input=False) -> '_BaseHistory':
        path = self._input_path if from_input else self._output_path
        try:
            with open(path, "rb") as f:
                history: _BaseHistory = pickle.load(f)
                self._history_list = history.list()
                return history
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No previous '{path}' history found, starting fresh.")

    @abstractmethod
    def plot_history(self, title: str = None) -> None:
        pass


class _BaseHistoryItem(_ConfigMixin, ABC):
    def __init__(self, epoch: int):
        super().__init__()
        self.epoch = epoch

    @abstractmethod
    def __str__(self) -> str:
        pass

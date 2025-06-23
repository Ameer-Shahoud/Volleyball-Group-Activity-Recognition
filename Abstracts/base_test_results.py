from abc import ABC, abstractmethod
import os
import pickle

from Abstracts.config_mixin import _ConfigMixin


class _BaseTestResults(_ConfigMixin, ABC):
    def __init__(
        self,
        input_path: str = None,
        suffix: str = None
    ):
        self._input_path = input_path if input_path else os.path.join(
            self.get_bl_cf().output_dir,
            f'test-results.{suffix}.pkl' if suffix else 'test-results.pkl'
        )

        self._output_path = os.path.join(
            self.get_bl_cf().output_dir,
            f'test-results.{suffix}.pkl' if suffix else 'test-results.pkl'
        )

        self._confusion_output_path = os.path.join(
            self.get_bl_cf().output_dir,
            f'test-results-confusion-matrix.{suffix}.png' if suffix else 'test-results-confusion-matrix.png'
        )

    def save(self):
        with open(self._output_path, "wb") as f:
            pickle.dump(self, f)

    def load(self, from_input=False) -> '_BaseTestResults':
        path = self._input_path if from_input else self._output_path
        try:
            with open(path, "rb") as f:
                test_results: _BaseTestResults = pickle.load(f)
                return test_results
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No previous '{path}' test results found.")

    @abstractmethod
    def plot_confustion_matrix(self, title: str = None) -> None:
        pass

    @abstractmethod
    def print_classification_report(self, title: str = None) -> None:
        pass

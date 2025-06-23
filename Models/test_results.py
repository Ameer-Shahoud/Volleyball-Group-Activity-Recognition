from matplotlib import pyplot as plt
import seaborn as sns
from Abstracts.base_test_results import _BaseTestResults

from Models.metrics import Metrics


class TestResults(_BaseTestResults):
    def __init__(
        self,
        test_metrics: Metrics,
        input_path: str = None,
        suffix: str = None,
        levels: list[str] = []
    ):
        super().__init__(input_path, suffix)
        self.levels = levels
        self.test_metrics = test_metrics

        self.get_bl_cf().logger.info(
            "Test Results:\n",
            *[f"{level.capitalize()} Level --- \t Loss: {self.test_metrics.get_loss(level):.4f} \t Acc: {self.test_metrics.get_acc(level):.2f}% \t F1-Score: {self.test_metrics.get_f1(level):.3f}" for level in self.levels]
        )

    def plot_confustion_matrix(self, write_fig_to_tensorboard=True):
        cm = [
            self.test_metrics.get_confusion_matrix(level) for level in self.levels
        ]

        levels_count = len(self.levels)

        fig, axes = plt.subplots(
            levels_count, 1, figsize=(8, 7 * levels_count)
        )

        if self.get_bl_cf().title:
            fig.suptitle(self.get_bl_cf().title, fontsize=16)

        for i, level in enumerate(self.levels):
            ax = axes[i] if levels_count > 1 else axes

            sns.heatmap(
                cm[i],  # Handle both torch.Tensor and numpy arrays
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=self.get_cf().dataset.get_categories(level),
                yticklabels=self.get_cf().dataset.get_categories(level),
                ax=ax  # Plot on the specified axis
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{level.capitalize()} Level Confusion Matrix")

        plt.tight_layout()
        plt.savefig(self._confusion_output_path)
        if write_fig_to_tensorboard:
            self.get_bl_cf().writer.add_figure("Confusion Matrix", fig)
        plt.show()

    def print_classification_report(self, write_report_to_tensorboard=True):
        report = [
            self.test_metrics.get_classification_report(level) for level in self.levels
        ]

        for i, level in enumerate(self.levels):
            title = f"-------- {level.capitalize()} Level - Classification Report --------"
            l = len(title)
            line = '\n' + '-' * l + '\n'
            if write_report_to_tensorboard:
                self.get_bl_cf().writer.add_text(
                    "Classification Report", line + title + line + report[i]
                )

            self.get_bl_cf().logger.info(
                line + title + line + report[i],
            )

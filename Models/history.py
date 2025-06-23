from Abstracts.base_history import _BaseHistory, _BaseHistoryItem
import matplotlib.pyplot as plt

from Models.metrics import Metrics


class History(_BaseHistory):
    def __init__(self, input_path: str = None, suffix: str = None):
        super().__init__(input_path, suffix)

    def plot_history(self, write_fig_to_tensorboard=True):
        if not len(self.list()):
            return

        items: list[HistoryItem] = self.list()
        levels = items[0].levels
        levels_count = len(items[0].levels)

        fig, axes = plt.subplots(
            levels_count, 3, figsize=(18, 4 * levels_count)
        )

        if self.get_bl_cf().title:
            fig.suptitle(self.get_bl_cf().title, fontsize=16)

        for i, level in enumerate(levels):
            ax_loss = axes[i][0] if levels_count > 1 else axes[0]
            ax_acc = axes[i][1] if levels_count > 1 else axes[1]
            ax_f1 = axes[i][2] if levels_count > 1 else axes[2]

            # Loss
            ax_loss.plot(
                list(map(lambda t: t.train_metrics.get_loss(level), items)),
                label='Train Loss'
            )
            ax_loss.plot(
                list(map(lambda t: t.val_metrics.get_loss(level), items)),
                label='Val Loss'
            )
            ax_loss.set_title(
                f'{level.capitalize()} Level - Loss'
            )
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()

            # Accuracy
            ax_acc.plot(
                list(map(lambda t: t.train_metrics.get_acc(level), items)),
                label='Train Acc'
            )
            ax_acc.plot(
                list(map(lambda t: t.val_metrics.get_acc(level), items)),
                label='Val Acc'
            )
            ax_acc.set_title(
                f'{level.capitalize()} Level - Accuracy'
            )
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.legend()

            # F1 Score
            ax_f1.plot(
                list(map(lambda t: t.train_metrics.get_f1(level), items)),
                label='Train F1-Score'
            )
            ax_f1.plot(
                list(map(lambda t: t.val_metrics.get_f1(level), items)),
                label='Val F1-Score'
            )
            ax_f1.set_title(
                f'{level.capitalize()} Level - F1-Score'
            )
            ax_f1.set_xlabel('Epoch')
            ax_f1.set_ylabel('F1-Score')
            ax_f1.legend()

        plt.tight_layout()
        plt.savefig(self._fig_output_path)
        plt.show()
        if write_fig_to_tensorboard:
            self.get_bl_cf().writer.add_figure("History figures", fig)


class HistoryItem(_BaseHistoryItem):
    def __init__(
        self,
        epoch: int,
        train_metrics: Metrics,
        val_metrics: Metrics,
        levels: list[str] = []
    ):
        super().__init__(epoch)
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.levels = levels

        for level in levels:
            self.get_bl_cf().writer.add_scalars(
                main_tag=f"{level.capitalize()}/Loss",
                tag_scalar_dict={
                    "Train": train_metrics.get_loss(level),
                    "Validation": val_metrics.get_loss(level),
                },
                global_step=epoch
            )
            self.get_bl_cf().writer.add_scalars(
                main_tag=f"{level.capitalize()}/Accuracy",
                tag_scalar_dict={
                    "Train": train_metrics.get_acc(level),
                    "Validation": val_metrics.get_acc(level),
                },
                global_step=epoch
            )
            self.get_bl_cf().writer.add_scalars(
                main_tag=f"{level.capitalize()}/F1-Score",
                tag_scalar_dict={
                    "Train": train_metrics.get_f1(level),
                    "Validation": val_metrics.get_f1(level),
                },
                global_step=epoch
            )

    def __str__(self):
        s = ''
        s += f"Training - Epoch[{self.epoch}/{self.get_bl_cf().training.epochs}]\n"
        for level in self.levels:
            s += f"{level.capitalize()} Level:\n--- Train --- \t\t Loss: {self.train_metrics.get_loss(level):.3f} \t Acc: {self.train_metrics.get_acc(level):.2f}% \t F1-Score: {self.train_metrics.get_f1(level):.3f} \n--- Validation --- \t Loss: {self.val_metrics.get_loss(level):.3f} \t Acc: {self.val_metrics.get_acc(level):.2f}% \t F1-Score: {self.val_metrics.get_f1(level):.3f}\n"
        return s

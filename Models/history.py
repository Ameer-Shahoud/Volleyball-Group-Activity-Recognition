from Abstracts.base_history import _BaseHistory, _BaseHistoryItem
import matplotlib.pyplot as plt


class History(_BaseHistory):
    def __init__(self, input_path: str = None, suffix: str = None):
        super().__init__(input_path, suffix)

    def plot_history(self):
        if not len(self.list()):
            return

        items: list[HistoryItem] = self.list()
        labels_count = len(items[0].labels)
        fig, axes = plt.subplots(
            labels_count, 2, figsize=(12, 4 * labels_count)
        )

        for l in range(labels_count):
            ax_loss = axes[l][0] if labels_count > 1 else axes[0]
            ax_acc = axes[l][1] if labels_count > 1 else axes[1]

            # Loss
            ax_loss.plot(
                list(map(lambda x: x[l], map(lambda t: t.train_loss, items))),
                label='Train Loss'
            )
            ax_loss.plot(
                list(map(lambda x: x[l], map(lambda t: t.val_loss, items))),
                label='Val Loss'
            )
            ax_loss.set_title(
                f'{items[0].labels[l].capitalize()} Level - Loss'
            )
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()

            # Accuracy
            ax_acc.plot(
                list(map(lambda x: x[l], map(lambda t: t.train_acc, items))),
                label='Train Acc'
            )
            ax_acc.plot(
                list(map(lambda x: x[l], map(lambda t: t.val_acc, items))),
                label='Val Acc'
            )
            ax_acc.set_title(
                f'{items[0].labels[l].capitalize()} Level - Accuracy'
            )
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.legend()

        plt.tight_layout()
        plt.show()


class HistoryItem(_BaseHistoryItem):
    def __init__(
        self,
        epoch: int,
        train_loss: list[float],
        train_acc: list[float],
        val_loss: list[float],
        val_acc: list[float],
        labels: list[str] = []
    ):
        super().__init__(epoch)
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.val_loss = val_loss
        self.val_acc = val_acc
        self.labels = labels

    def to_dict(self) -> dict[str, object]:
        return {
            'epoch': self.epoch,
            'train-loss': self.train_loss,
            'train-acc': self.train_acc,
            'val-loss': self.val_loss,
            'val-acc': self.val_acc,
        }

    def __str__(self):
        s = ''
        for i in range(len(self.labels)):
            s += f"{self.labels[i].capitalize()}  ---  Train Loss: {self.train_loss[i]:.3f} - Train Acc: {self.train_acc[i]:.2f}% - Val Loss: {self.val_loss[i]:.3f} - Val Acc: {self.val_acc[i]:.2f}%\n"
        return s

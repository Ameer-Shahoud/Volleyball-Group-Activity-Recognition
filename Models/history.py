from Abstracts.base_history import _BaseHistory, _BaseHistoryItem


class History(_BaseHistory):
    def __init__(self, input_path: str = None, suffix: str = None):
        super().__init__(input_path, suffix)

    def plot_history(self):
        pass


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

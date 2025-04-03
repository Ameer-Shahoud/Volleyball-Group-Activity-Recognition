from Models.base_history import _BaseHistory, _BaseHistoryItem


class B2JointHistory(_BaseHistory):
    def __init__(self, input_path: str = None):
        super().__init__(input_path)

    def plot_history(self):
        pass


class B2JointHistoryItem(_BaseHistoryItem):
    def __init__(
        self,
        epoch: int,
        player_train_loss: float,
        player_train_acc: float,
        player_val_loss: float,
        player_val_acc: float,
        img_train_loss: float,
        img_train_acc: float,
        img_val_loss: float,
        img_val_acc: float,
    ):
        super().__init__(epoch)
        self.player_train_loss = player_train_loss
        self.player_train_acc = player_train_acc
        self.player_val_loss = player_val_loss
        self.player_val_acc = player_val_acc

        self.img_train_loss = img_train_loss
        self.img_train_acc = img_train_acc
        self.img_val_loss = img_val_loss
        self.img_val_acc = img_val_acc

    def to_dict(self) -> dict[str, object]:
        return {
            'epoch': self.epoch,
            'player-train-loss': self.player_train_loss,
            'player-train-acc': self.player_train_acc,
            'player-val-loss': self.player_val_loss,
            'player-val-acc': self.player_val_acc,
            'img-train-loss': self.img_train_loss,
            'img-train-acc': self.img_train_acc,
            'img-val-loss': self.img_val_loss,
            'img-val-acc': self.img_val_acc,
        }

    def __str__(self):
        return f"""
            Player Train Loss: {self.player_train_loss:.3f} - Player Train Acc: {self.player_train_acc:.2f}% - Player Val Loss: {self.player_val_loss:.3f} - Player Val Acc: {self.player_val_acc:.2f}%
            Image Train Loss: {self.img_train_loss:.3f} - Image Train Acc: {self.img_train_acc:.2f}% - Image Val Loss: {self.img_val_loss:.3f} - Image Val Acc: {self.img_val_acc:.2f}%
        """

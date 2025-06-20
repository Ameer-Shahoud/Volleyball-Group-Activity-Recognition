from torch import nn


class ClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        num_classes: int = None,
        with_batch_norm=True,
        with_dropout=True,
        dropout_rate=0.5
    ):
        super().__init__()

        self.head = nn.Sequential(
            *[
                nn.Linear(input_dim, hidden_dim),
                *([nn.BatchNorm1d(hidden_dim)] if with_batch_norm else []),
                nn.ReLU(),
                *([nn.Dropout(dropout_rate)] if with_dropout else []),
                nn.Linear(hidden_dim, num_classes)
            ]
        )

    def forward(self, x):
        return self.head(x)

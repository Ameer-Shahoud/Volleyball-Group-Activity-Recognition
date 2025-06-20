from torch import nn
from Modules.classifier_head import ClassifierHead
from Modules.custom_max_pool import CustomMaxPool


class CropsClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        num_classes: int = None,
        pool_dim=1,
        with_batch_norm=True,
        with_dropout=True,
        dropout_rate=0.5
    ):
        super().__init__()

        self.head = nn.Sequential(
            *[
                CustomMaxPool(dim=pool_dim),
                ClassifierHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    with_batch_norm=with_batch_norm,
                    with_dropout=with_dropout,
                    dropout_rate=dropout_rate,
                    num_classes=num_classes
                ),
            ]
        )

    def forward(self, x):
        return self.head(x)

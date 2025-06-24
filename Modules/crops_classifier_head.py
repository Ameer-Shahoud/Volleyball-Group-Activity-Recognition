from torch import nn
from Modules.classifier_head import ClassifierHead
from Modules.custom_max_pool import CustomMaxPool


class CropsClassifierHead(ClassifierHead):
    def __init__(
        self,
        pool_dim=1,
        input_dim=2048,
        hidden_dim=512,
        num_classes: int = None,
        with_batch_norm=False,
        with_dropout=False,
        dropout_rate=0.5
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            with_batch_norm=with_batch_norm,
            with_dropout=with_dropout,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )

        self.pool = CustomMaxPool(dim=pool_dim)

    def forward(self, x):
        x = self.pool(x)
        return self.head(x)

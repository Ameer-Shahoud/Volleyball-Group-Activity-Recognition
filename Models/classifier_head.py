from torch import nn

from Models.custom_max_pool import CustomMaxPool


class ClassifierHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_classes: int = None, pool_dim=1):
        super().__init__()

        self.head = nn.Sequential(
            CustomMaxPool(dim=pool_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)

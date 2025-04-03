import torch
import torch.nn as nn


class CustomMaxPool(nn.Module):
    def __init__(self, dim: int,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dim = dim

    def forward(self, x):
        # (batch_size, num_players, 2048) -> (batch_size, 2048)
        return torch.max(x, dim=self.__dim, keepdim=False)[0]

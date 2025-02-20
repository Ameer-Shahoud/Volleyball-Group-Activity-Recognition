import torch
from torch import nn
from torch.utils.data import DataLoader
from Enums.classification_level import ClassificationLevel
from Models.config_mixin import _ConfigMixin
from Utils.cuda import get_device


class ModelBase(nn.Module, _ConfigMixin):
    def __init__(self, backbone: nn.Module, level: ClassificationLevel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = backbone
        self.__in_features = self.base_model.fc.in_features
        self.classifier = nn.Linear(
            self.__in_features,
            len(self.get_cf().dataset.get_categories(level))
        )
        self.base_model.fc = nn.Identity()

    def forward(self, x: torch.Tensor, return_features=False):
        x = x.to(get_device())
        features = self.base_model(x)

        if return_features:
            return features.to(get_device())

        return self.classifier(features).to(get_device())

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        pass

    def set_backbone_requires_grad(self, requires_grad: bool):
        for param in self.base_model.parameters():
            param.requires_grad = requires_grad
        return self

    def set_backbone_layer_requires_grad(self, layer: str, requires_grad: bool):
        for name, param in self.base_model.named_parameters():
            if layer in name:
                param.requires_grad = requires_grad
        return self

    def to_available_device(self):
        self.to(get_device())
        return self

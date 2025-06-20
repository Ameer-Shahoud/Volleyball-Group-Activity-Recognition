import torch
from torch import nn
from Enums.classification_level import ClassificationLevel
from Abstracts.config_mixin import _ConfigMixin
from Utils.cuda import get_device
from torchvision import models


class BackboneModel(nn.Module, _ConfigMixin):
    def __init__(self, level: ClassificationLevel, backbone: nn.Module = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone if backbone else models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        self.__in_features = self.backbone.fc.in_features
        self.classifier = nn.Linear(
            self.__in_features,
            len(self.get_cf().dataset.get_categories(level))
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor):
        batch_size, frames_count, channels, width, height = x.shape
        x = x.view(-1, channels, width, height)
        x = x.to(get_device())

        features = self.backbone(x)

        return self.classifier(features).view(batch_size, frames_count, -1).to(get_device()), features.view(batch_size, frames_count, -1).to(get_device())

    def set_backbone_requires_grad(self, requires_grad: bool):
        for param in self.backbone.parameters():
            param.requires_grad = requires_grad
        return self

    def set_backbone_layer_requires_grad(self, layer: str, requires_grad: bool):
        for name, param in self.backbone.named_parameters():
            if layer in name:
                param.requires_grad = requires_grad
        return self

    def set_all_requires_grad(self, requires_grad: bool):
        for param in self.classifier.parameters():
            param.requires_grad = requires_grad
        return self.set_backbone_requires_grad(requires_grad=requires_grad)

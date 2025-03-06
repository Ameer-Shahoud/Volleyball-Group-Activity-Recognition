import torch
from torch import nn
from torch.utils.data import DataLoader
from Enums.classification_level import ClassificationLevel
from Models.config_mixin import _ConfigMixin
from Utils.cuda import get_device


class BaseModel(nn.Module, _ConfigMixin):
    """
    Base class for building custom models using pre-trained backbones.
    It wraps a pre-trained model and customizes the classifier layer for the given task.

    Attributes:
        base_model (nn.Module): Backbone model for feature extraction.
        classifier (nn.Linear): Custom classification layer.
    """

    def __init__(self, backbone: nn.Module, level: ClassificationLevel, *args, **kwargs):
        """
        Initializes the ModelBase with a pre-trained backbone.

        Args:
            backbone (nn.Module): Pre-trained model for feature extraction.
            level (ClassificationLevel): Classification level (IMAGE or PLAYER).
        """
        super().__init__(*args, **kwargs)
        self.base_model = backbone
        self.__in_features = self.base_model.fc.in_features
        self.classifier = nn.Linear(
            self.__in_features,
            len(self.get_cf().dataset.get_categories(level))
        )
        self.base_model.fc = nn.Identity()

    def forward(self, x: torch.Tensor, return_features=False):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.
            return_features (bool): If True, return feature vectors instead of predictions.

        Returns:
            torch.Tensor: Model output or feature vectors.
        """
        batch_size, frames_count, channels, width, height = x.shape
        x = x.view(-1, channels, width, height)
        x = x.to(get_device())

        features = self.base_model(x)

        if return_features:
            return features.view(batch_size, frames_count, -1).to(get_device())

        return self.classifier(features).view(batch_size, frames_count, -1).to(get_device())

    def set_backbone_requires_grad(self, requires_grad: bool):
        """
        Sets the requires_grad attribute for the backbone model.

        Args:
            requires_grad (bool): True to enable gradient computation, False to freeze layers.
        """
        for param in self.base_model.parameters():
            param.requires_grad = requires_grad
        return self

    def set_backbone_layer_requires_grad(self, layer: str, requires_grad: bool):
        """
        Sets the requires_grad attribute for a specific layer in the backbone.

        Args:
            layer (str): Layer name to be set.
            requires_grad (bool): True to enable gradient computation, False to freeze the layer.
        """
        for name, param in self.base_model.named_parameters():
            if layer in name:
                param.requires_grad = requires_grad
        return self

    def to_available_device(self):
        """Transfers the model to the available device (CPU or GPU)."""
        self.base_model = self.base_model.to(get_device())
        self.classifier = self.classifier.to(get_device())
        return self

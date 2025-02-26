from Models.training_base import _TrainingBase
from Enums.classification_level import ClassificationLevel
from Enums.dataset_type import DatasetType
from Models.model_base import ModelBase
from Models.dataset import ImageDataset
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class TrainingB1(_TrainingBase):
    """
    Training pipeline for the B1 baseline model.
    This class extends _TrainingBase and implements specific logic for:
    - Data loading
    - Model preparation
    - Optimizer and scheduler configuration
    - Training loop
    - Evaluation and testing

    Attributes:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        model (ModelBase): Model used for training and evaluation.
        criterion (nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    """
    def _prepare_loaders(self):
        """
        Prepares DataLoaders for training, validation, and testing.

        - Loads ImageDataset for train, val, and test sets.
        - Initializes DataLoaders with the configured batch size."""
        train_dataset = ImageDataset(type=DatasetType.TRAIN)
        val_dataset = ImageDataset(type=DatasetType.VAL)
        test_dataset = ImageDataset(type=DatasetType.TEST)

        batch_size = self.get_bl_cf().training.batch_size

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

    def _prepare_model(self):
        """
        Prepares the model for training.

        - Loads a pre-trained ResNet50 model.
        - Sets the classification level to IMAGE.
        - Freezes all layers except 'layer4' and 'fc'.
        - Initializes a custom classifier layer for group activity classification.
        """
        self.model = ModelBase(
            backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            level=ClassificationLevel.IMAGE
        ) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

    def _prepare_optimizer(self):
        """
        Prepares the optimizer and learning rate scheduler.

        - Uses Adam optimizer for updating model weights.
        - Initializes a learning rate scheduler (ReduceLROnPlateau).  
        """
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )

    def _get_train_loader(self):
        """
        Retrieves the DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return self.train_loader

    def _get_val_loader(self):
        """
        Retrieves the DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        return self.val_loader

    def _get_test_loader(self):
        """
        Retrieves the DataLoader for testing data.

        Returns:
            DataLoader: DataLoader for testing data.
        """
        return self.test_loader

    def _train_mode(self):
        """Sets the model to training mode."""
        self.model.train()

    def _eval_mode(self):
        """Sets the model to evaluation mode."""
        self.model.eval()

    def _on_epoch_step(self):
        """Updates the learning rate scheduler at the end of each epoch."""
        self.scheduler.step(self.val_loss)

    def _train_step(self, inputs, labels):
        """
        Defines a single training step.

        - Clears the gradients.
        - Performs a forward pass to get predictions.
        - Computes the loss using CrossEntropyLoss.
        - Backpropagates the loss and updates the model weights.
        - Calculates training accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        self.loss += loss.item()

        _, predicted = outputs.max(1)
        self.correct += (predicted == labels).sum().item()
        self.total += labels.size(0)

    def _eval_step(self, inputs, labels):
        """
        Defines a single evaluation step.

        - Performs a forward pass to get predictions.
        - Computes the loss using CrossEntropyLoss.
        - Calculates validation accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.val_loss += loss.item()

        _, predicted = outputs.max(1)
        self.val_correct += (predicted == labels).sum().item()
        self.val_total += labels.size(0)

    def _test_step(self, inputs, labels):
        """
        Defines a single testing step.

        - Performs a forward pass to get predictions.
        - Computes the loss using CrossEntropyLoss.
        - Calculates test accuracy.

        Args:
            inputs (torch.Tensor): Input images.
            labels (torch.Tensor): Ground truth labels.
        """
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.test_loss += loss.item()

        _, predicted = outputs.max(1)
        self.test_correct += (predicted == labels).sum().item()
        self.test_total += labels.size(0)

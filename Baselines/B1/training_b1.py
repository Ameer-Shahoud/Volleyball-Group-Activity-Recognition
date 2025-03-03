import torch
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

from Utils import cuda


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
        self.model = ModelBase(
            backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            level=ClassificationLevel.IMAGE
        ) \
            .set_backbone_requires_grad(False) \
            .set_backbone_layer_requires_grad('layer4', True) \
            .set_backbone_layer_requires_grad('fc', True)

    def _prepare_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.get_bl_cf().training.learning_rate
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=2
        )

    def _to_available_device(self):
        self.model.to(cuda.get_device())
        for state in self.optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(cuda.get_device())
        self.scheduler.st

    def _get_train_loader(self):
        return self.train_loader

    def _get_val_loader(self):
        return self.val_loader

    def _get_test_loader(self):
        return self.test_loader

    def _train_mode(self):
        self.model.train()

    def _eval_mode(self):
        self.model.eval()

    def _load_last_checkpoint(self) -> int:
        checkpoint = torch.load(self._checkpoint_path,
                                map_location=cuda.get_device())
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._to_available_device()
        return checkpoint['epoch']

    def _on_epoch_step(self, epoch: int):
        self.scheduler.step(self.val_loss)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        },  self._checkpoint_path)

    def _save_trained_model(self):
        torch.save(self.model, self._model_path)

    def _train_step(self, inputs, labels):
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
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.val_loss += loss.item()

        _, predicted = outputs.max(1)
        self.val_correct += (predicted == labels).sum().item()
        self.val_total += labels.size(0)

    def _test_step(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.test_loss += loss.item()

        _, predicted = outputs.max(1)
        self.test_correct += (predicted == labels).sum().item()
        self.test_total += labels.size(0)

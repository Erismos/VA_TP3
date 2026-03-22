import torch
import torch.nn as nn
from torchvision import models


class TransferModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Remplacer la couche finale
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def unfreeze_last_layers(model, num_layers=10):
    """Unfreeze last layers for fine-tuning"""
    layers = list(model.model.children())

    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    return model
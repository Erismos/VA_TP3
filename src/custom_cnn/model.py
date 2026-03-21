"""Simple and explicit CNN from scratch for butterfly classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ButterflyCNN(nn.Module):
    """Simple CNN with explicit layer-by-layer definition."""

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        # Block 1: 3 -> 64, then pool (224 -> 112)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64 -> 128, then pool (112 -> 56)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 128 -> 256, then pool (56 -> 28)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 256 -> 512, then pool (28 -> 14)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 512 -> 512 without pooling (keep 14x14)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 1024)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

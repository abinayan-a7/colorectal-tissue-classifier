"""Custom 4-block CNN architecture (HistoCNN) for colorectal tissue classification."""

import torch
import torch.nn as nn


class HistoCNN(nn.Module):
    """Custom 4-block Convolutional Neural Network (~308k parameters).

    Follows the CNN funnel pattern for 64x64 H&E histopathology patches:
      - Block 1: 3  -> 32  channels, MaxPool (64x64 -> 32x32)
      - Block 2: 32 -> 64  channels, MaxPool (32x32 -> 16x16)
      - Block 3: 64 -> 128 channels, MaxPool (16x16 -> 8x8)
      - Block 4: 128 -> 128 channels, AdaptiveAvgPool (8x8 -> 2x2)
      - Classifier: Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, num_classes)
      - Classifier: Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, num_classes)

    Outputs raw unnormalized logits of shape (N, num_classes).
    """

    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4: 8x8 -> 2x2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw logits for input batch x."""
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int = 9) -> nn.Module:
    """Instantiate and return HistoCNN with the specified number of classes."""
    return HistoCNN(num_classes=num_classes)

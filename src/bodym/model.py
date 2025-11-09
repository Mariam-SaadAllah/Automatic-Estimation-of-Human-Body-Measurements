from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class MNASNetRegressor(nn.Module):
    """
    A regression model based on MNASNet (Mobile Neural Architecture Search network).
    Initializes an MNASNet backbone (1.0) with optional ImageNet pre-trained weights,
    then replaces the classifier to output `num_outputs` continuous values.

    Outputs remain normalized in [-1, +1] via Tanh(), exactly as before.
    Compatible with evaluation scripts that convert predictions back to millimeters.
    """

    def __init__(self, num_outputs: int = 14, weights: str | None = "IMAGENET1K_V1"):
        super().__init__()

     
        # Load MNASNet with or without pretrained ImageNet weights
      
        if weights is None or weights.upper() == "NONE":
            mnas = models.mnasnet1_0(weights=None)
        else:
            enum = getattr(models, "MNASNet1_0_Weights")[weights]
            mnas = models.mnasnet1_0(weights=enum)

       
        # Keep the convolutional backbone (feature extractor)
        # We'll explicitly store it in self.backbone for optimizer access.
        self.backbone = mnas.layers  # feature extraction layers

        # Replace the final classifier with a regression head
        num_features = mnas.classifier[1].in_features  # 1280 for MNASNet1_0

        # new fully-connected regression head
        self.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
            nn.Tanh(),  # keep outputs in [-1, +1]
        )

        # Keep backward compatibility with older scripts (e.g., train.py)
        self.classifier = self.fc
        
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MNASNet feature extractor expects flattened avgpool output
        x = self.backbone(x)                 # convolutional feature maps
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # global avg pool
        x = torch.flatten(x, 1)              # flatten before FC layers
        x = self.fc(x)                       # regression head
        return x









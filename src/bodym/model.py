from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

class MNASNetRegressor(nn.Module):
    """
    A regression model based on MNASNet (Mobile Neural Architecture Search network).
    Initializes an MNASNet backbone (1.0) with optional ImageNet pre-trained weights, 
    then replaces the classifier to output `num_outputs` continuous values.
    """
    def __init__(self, num_outputs: int = 14, weights: str | None = "IMAGENET1K_V1"):
        super().__init__()
        if weights is None or weights.upper() == "NONE":
            # Initialize model without pre-trained weights.
            mnas = models.mnasnet1_0(weights=None)
        else:
            # Load pre-trained weights from torchvision (ImageNet).
            enum = getattr(models, "MNASNet1_0_Weights")[weights]
            mnas = models.mnasnet1_0(weights=enum)
        # Replace the final classifier layers.
        num_features = mnas.classifier[1].in_features  # features of last linear layer
        mnas.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )
        self.model = mnas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)




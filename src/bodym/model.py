from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

class MNASNetRegressor(nn.Module):
    """
    A regression model based on MNASNet (Mobile Neural Architecture Search network).
    Uses an MNASNet backbone (1.0) with optional ImageNet pre-trained weights, and 
    replaces the classifier to output `num_outputs` continuous values in range [-1, +1].
    """
    def __init__(self, num_outputs: int = 14, weights: str | None = "IMAGENET1K_V1"):
        super().__init__()
        
        # -------------------------------------------------------------
        # 1. Load MNASNet backbone (with or without ImageNet weights)
        # -------------------------------------------------------------
        if weights is None or weights.upper() == "NONE":
            mnas = models.mnasnet1_0(weights=None)
        else:
            enum = getattr(models, "MNASNet1_0_Weights")[weights]
            mnas = models.mnasnet1_0(weights=enum)

        # -------------------------------------------------------------
        # 2. Replace the final classifier
        # -------------------------------------------------------------
        num_features = mnas.classifier[1].in_features  # input features of last linear layer
        mnas.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),           # non-linearity in hidden layer
            nn.Linear(128, num_outputs),
            nn.Tanh(),           # bound outputs to [-1, +1]
        )

        # -------------------------------------------------------------
        # 3. Store model
        # -------------------------------------------------------------
        self.model = mnas

    # -------------------------------------------------------------
    # 4. Forward pass
    # -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regression.
        Returns normalized predictions in range [-1, +1].
        """
        return self.model(x)



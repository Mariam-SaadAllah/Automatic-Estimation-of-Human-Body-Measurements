"""
model.py
------------------------------------------------------------
Defines the MNASNet-based regression model (BMnet) used for 
body measurement estimation from silhouette images.

This model follows Section 3.1 of the paper:
    "Human Body Measurement Estimation with Adversarial Augmentation"

Architecture summary:
  - MNASNet 1.0 backbone (from torchvision)
  - Optional ImageNet pretraining
  - Final regression head: 128 → 14 outputs
  - Tanh activation ensures output range [-1, +1]

Each output corresponds to one normalized body measurement
(ankle, arm-length, bicep, calf, chest, ... height).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models


class MNASNetRegressor(nn.Module):
    """
    MNASNet-based regressor for 14 body measurements.
    
    Args:
        num_outputs (int): Number of continuous measurement outputs (default=14)
        weights (str | None): 
            - "IMAGENET1K_V1" → load ImageNet pretrained weights
            - "NONE" or None → random initialization
    """
    def __init__(self, num_outputs: int = 14, weights: str | None = "IMAGENET1K_V1"):
        super().__init__()

        # ----------------------------------------------------------
        # 1) Load MNASNet backbone
        # ----------------------------------------------------------
        if weights is None or weights.upper() == "NONE":
            # Random initialization
            mnas = models.mnasnet1_0(weights=None)
        else:
            # Pre-trained ImageNet weights
            enum = getattr(models, "MNASNet1_0_Weights")[weights]
            mnas = models.mnasnet1_0(weights=enum)

        # ----------------------------------------------------------
        # 2) Replace classifier with regression head
        # ----------------------------------------------------------
        num_features = mnas.classifier[1].in_features  # last linear layer input
        mnas.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs),
            nn.Tanh(),  # outputs ∈ [-1, +1] after normalization
        )

        self.model = mnas

    # --------------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BMnet regression.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, 2W)
                              (concatenated frontal + lateral silhouettes)
        Returns:
            torch.Tensor: 14 continuous measurement predictions in [-1, +1]
        """
        return self.model(x)


# --------------------------------------------------------------
# (Optional) Quick test
# --------------------------------------------------------------
if __name__ == "__main__":
    model = MNASNetRegressor(num_outputs=14, weights="NONE")
    dummy = torch.randn(2, 3, 640, 960)  # batch of 2 subjects
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Output range: min={out.min().item():.3f}, max={out.max().item():.3f}")







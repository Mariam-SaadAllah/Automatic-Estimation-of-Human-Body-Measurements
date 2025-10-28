"""
Compute normalization constants (min/max) from the training split of BodyM dataset.
Run this ONCE in Colab to obtain height/weight and per-measurement min/max values.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
# Update this path to your training split folder
TRAIN_SPLIT_DIR = Path("/content/drive/MyDrive/BMNet_Project/data/train")

# List of measurement columns
MEASUREMENT_COLS = [
    "ankle",
    "arm-length",
    "bicep",
    "calf",
    "chest",
    "forearm",
    "hip",
    "leg-length",
    "shoulder-breadth",
    "shoulder-to-crotch",
    "thigh",
    "waist",
    "wrist",
    "height",
]

# -------------------------------
# Step 1: Load CSVs
# -------------------------------
measurements_df = pd.read_csv(TRAIN_SPLIT_DIR / "measurements.csv")
hwg_meta_df = pd.read_csv(TRAIN_SPLIT_DIR / "hwg_metadata.csv")

# -------------------------------
# Step 2: Compute height/weight min/max
# -------------------------------
height_min = hwg_meta_df["height_cm"].min()
height_max = hwg_meta_df["height_cm"].max()
weight_min = hwg_meta_df["weight_kg"].min()
weight_max = hwg_meta_df["weight_kg"].max()

# -------------------------------
# Step 3: Compute per-measurement min/max (convert cm → mm)
# -------------------------------
y_min_cm = measurements_df[MEASUREMENT_COLS].min(axis=0).values
y_max_cm = measurements_df[MEASUREMENT_COLS].max(axis=0).values
y_min_mm = y_min_cm * 10.0
y_max_mm = y_max_cm * 10.0

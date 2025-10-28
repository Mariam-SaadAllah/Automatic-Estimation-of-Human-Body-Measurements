from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import json

# ----------------------------------------------------------------------
# LOAD NORMALIZATION CONSTANTS (TRAINING SPLIT ONLY)
# ----------------------------------------------------------------------
NORM_PATH = Path("/content/drive/MyDrive/BMNet_Project/data/norm_constants.json")

if not NORM_PATH.exists():
    raise FileNotFoundError(
        f"❌ Normalization constants file not found at {NORM_PATH}.\n"
        "Please run compute_norm_stats.py first to generate it."
    )

with open(NORM_PATH, "r") as f:
    constants = json.load(f)

HEIGHT_MIN = float(constants["HEIGHT_MIN"])
HEIGHT_MAX = float(constants["HEIGHT_MAX"])
WEIGHT_MIN = float(constants["WEIGHT_MIN"])
WEIGHT_MAX = float(constants["WEIGHT_MAX"])
Y_MIN = np.array(constants["Y_MIN"], dtype=np.float32)
Y_MAX = np.array(constants["Y_MAX"], dtype=np.float32)

print(f"✅ Loaded normalization constants from {NORM_PATH}")

# ----------------------------------------------------------------------
# MEASUREMENT COLUMN NAMES (as provided by the dataset)
# ----------------------------------------------------------------------
MEASUREMENT_COLS: list[str] = [
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


# ----------------------------------------------------------------------
# DATA STRUCTURE FOR ONE SAMPLE
# ----------------------------------------------------------------------
@dataclass
class Sample:
    photo_id: str
    frontal: str
    lateral: str
    height: float       # normalized [0,1]
    weight: float       # normalized [0,1]
    height_cm: float
    weight_kg: float
    subject_id: str
    y_mm: list[float]   # 14 body measurements in mm
    y_norm: list[float] # normalized targets [-1,+1]


# ----------------------------------------------------------------------
# FUNCTION TO BUILD SAMPLES
# ----------------------------------------------------------------------
def build_samples(split_dir: Path) -> list[dict[str, Any]]:
    """
    Builds a list of samples for the given dataset split.
    Normalizes using precomputed training-split constants from JSON.
    """
    measurements_df = pd.read_csv(split_dir / "measurements.csv")
    subject_map_df = pd.read_csv(split_dir / "subject_to_photo_map.csv")
    hwg_meta_df = pd.read_csv(split_dir / "hwg_metadata.csv")

    mask_dir = split_dir / "mask"
    mask_left_dir = split_dir / "mask_left"

    subj_to_hw = {row["subject_id"]: row for _, row in hwg_meta_df.iterrows()}
    subj_to_measure = {row["subject_id"]: row for _, row in measurements_df.iterrows()}
    subj_to_pid = {row["subject_id"]: row["photo_id"] for _, row in subject_map_df.iterrows()}

    samples: list[dict[str, Any]] = []

    for subj, pid in subj_to_pid.items():
        front_path = mask_dir / f"{pid}.png"
        side_path = mask_left_dir / f"{pid}.png"
        if not front_path.exists() or not side_path.exists():
            continue

        hw = subj_to_hw.get(subj)
        meas_row = subj_to_measure.get(subj)
        if hw is None or meas_row is None:
            continue

        # Raw height & weight
        height_cm = float(hw["height_cm"])
        weight_kg = float(hw["weight_kg"])

        # Normalize height & weight
        height_norm = (height_cm - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN)
        weight_norm = (weight_kg - WEIGHT_MIN) / (WEIGHT_MAX - WEIGHT_MIN)

        # Convert 14 body measurements (cm → mm)
        y_cm = np.array([float(meas_row[c]) for c in MEASUREMENT_COLS], dtype=np.float32)
        y_mm = y_cm * 10.0

        # Normalize to [-1, +1]
        y_norm = 2 * ((y_mm - Y_MIN) / (Y_MAX - Y_MIN)) - 1
        y_norm = np.clip(y_norm, -1, 1)

        samples.append({
            "photo_id": pid,
            "frontal": str(front_path),
            "lateral": str(side_path),
            "height": height_norm,
            "weight": weight_norm,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "subject_id": subj,
            "y_mm": y_mm.tolist(),
            "y_norm": y_norm.tolist(),
        })

    return samples


# ----------------------------------------------------------------------
# BODYM DATASET CLASS
# ----------------------------------------------------------------------
class BodyMDataset(Dataset):
    """
    PyTorch Dataset for BodyM silhouettes and body measurements.

    Returns:
      - x: Tensor(3, H, 2W)  → [silhouette, height_map, weight_map]
      - y: Tensor(14,)       → normalized targets [-1, +1]
      - subject_id: str
    """
    def __init__(self, samples: list[dict[str, Any]], single_h: int = 640, single_w: int = 480, debug: bool = False):
        self.samples = samples
        self.single_h = single_h
        self.single_w = single_w
        self.debug = debug

        # ImageNet normalization for 3 channels
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_resize(self, path: str) -> np.ndarray:
        """Load grayscale silhouette and resize."""
        img = Image.open(path).convert("L")
        img = img.resize((self.single_w, self.single_h), resample=Image.NEAREST)
        return np.array(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Load both silhouettes
        front = self._load_and_resize(s["frontal"])
        side = self._load_and_resize(s["lateral"])
        concat_img = np.concatenate([front, side], axis=1).astype(np.float32)

        # Height and weight maps
        h_map = np.full_like(concat_img, s["height"], dtype=np.float32)
        w_map = np.full_like(concat_img, s["weight"], dtype=np.float32)

        # Stack into 3-channel tensor
        stacked = np.stack([concat_img, h_map, w_map], axis=0)
        stacked = (stacked - self.imagenet_mean[:, None, None]) / self.imagenet_std[:, None, None]
        x = torch.from_numpy(stacked).float()

        # Normalized targets
        y_norm = torch.tensor(s["y_norm"], dtype=torch.float32)
        subject_id = s["subject_id"]

        return x, y_norm, subject_id

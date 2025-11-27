from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Precomputed normalization constants (from BodyM training split)
# --------------------------------------------------------------------
# Replace the placeholder values with the actual min/max you compute once
HEIGHT_MIN_CM = 152.3
HEIGHT_MAX_CM = 197.5
WEIGHT_MIN_KG = 46.1
WEIGHT_MAX_KG = 108.2

# Per-measurement min and max values (in millimeters)
Y_MIN_MM = np.array([
    148.5, 398.2, 204.3, 252.1, 692.7, 211.6, 711.2, 702.9,
    306.4, 703.2, 354.7, 602.5, 151.4, 1512.3
], dtype=np.float32)
Y_MAX_MM = np.array([
    250.1, 802.8, 405.9, 452.3, 1198.4, 403.1, 1102.0, 1199.7,
    502.5, 1203.9, 604.8, 999.6, 249.1, 2001.8
], dtype=np.float32)

# List of measurement column names as provided by the dataset (in centimeters).
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

@dataclass
class Sample:
    """Data class to hold information for one data sample (one subject/photo entry)."""
    photo_id: str
    frontal: str
    lateral: str
    height: float   # normalized height (0..1)
    weight: float   # normalized weight (0..1)
    height_cm: float
    weight_kg: float
    subject_id: str
    y_cm: list[float]  # target measurements in cm
    y: list[float]     # (alias of y_cm for compatibility)

def build_samples(split_dir: Path) -> list[dict[str, Any]]:
    """
    Read dataset CSV files from the given split directory (e.g., train, testA, testB)
    and build a list of sample dictionaries.
    """
    measurements_df = pd.read_csv(split_dir / "measurements.csv")
    subject_map_df = pd.read_csv(split_dir / "subject_to_photo_map.csv")
    hwg_meta_df = pd.read_csv(split_dir / "hwg_metadata.csv")

    mask_dir = split_dir / "mask"
    mask_left_dir = split_dir / "mask_left"

    # Build dictionaries for quick lookup by subject_id.
    subj_to_hw = {row["subject_id"]: row for _, row in hwg_meta_df.iterrows()}
    subj_to_measure = {row["subject_id"]: row for _, row in measurements_df.iterrows()}

    samples: list[dict[str, Any]] = []

    # iterate through ALL rows (subject_id, photo_id)
    for _, row in subject_map_df.iterrows():

        subj = row["subject_id"]
        pid = row["photo_id"]

        # Load *all* matching silhouette files
        frontal_list = sorted(mask_dir.glob(f"{pid}*.png"))
        lateral_list = sorted(mask_left_dir.glob(f"{pid}*.png"))

        if len(frontal_list) == 0 or len(lateral_list) == 0:
            continue

        num_pairs = min(len(frontal_list), len(lateral_list))

        hw = subj_to_hw.get(subj)
        meas_row = subj_to_measure.get(subj)
        if hw is None or meas_row is None:
            continue

        height_cm = float(hw["height_cm"])
        weight_kg = float(hw["weight_kg"])

        height_norm = (height_cm - HEIGHT_MIN_CM) / (HEIGHT_MAX_CM - HEIGHT_MIN_CM)
        weight_norm = (weight_kg - WEIGHT_MIN_KG) / (WEIGHT_MAX_KG - WEIGHT_MIN_KG)

        y_cm = [float(meas_row[c]) for c in MEASUREMENT_COLS]

        # create one sample per silhouette pair
        for i in range(num_pairs):
            samples.append({
                "photo_id": f"{pid}_{i}",
                "frontal": str(frontal_list[i]),
                "lateral": str(lateral_list[i]),
                "height": height_norm,
                "weight": weight_norm,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "subject_id": subj,
                "y_cm": y_cm,
                "y": y_cm,
            })

    return samples


class BodyMDataset(Dataset):
    """
    PyTorch Dataset for BodyM silhouette data.
    """
    def __init__(self, samples: list[dict[str, Any]], single_h: int = 640, single_w: int = 480, debug: bool = False):
        self.samples = samples
        self.single_h = single_h
        self.single_w = single_w
        self.debug = debug

        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_resize(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        img = img.resize((self.single_w, self.single_h), resample=Image.NEAREST)
        return np.array(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        front_arr = self._load_and_resize(s["frontal"])
        side_arr = self._load_and_resize(s["lateral"])
        concat_img = np.concatenate([front_arr, side_arr], axis=1).astype(np.float32)

        height_map = np.full_like(concat_img, fill_value=s["height"], dtype=np.float32)
        weight_map = np.full_like(concat_img, fill_value=s["weight"], dtype=np.float32)

        stacked = np.stack([concat_img, height_map, weight_map], axis=0)

        stacked = (stacked - self.imagenet_mean[:, None, None]) / self.imagenet_std[:, None, None]
        x = torch.from_numpy(stacked).float()

        # Prepare target measurements in millimeters
        y_cm = np.array(s["y_cm"], dtype=np.float32)
        y_mm = y_cm * 10.0  # convert cm → mm

        # Normalize to [-1, +1]
        y_norm = 2.0 * (y_mm - Y_MIN_MM) / (Y_MAX_MM - Y_MIN_MM) - 1.0

        y_tensor = torch.from_numpy(y_norm).float()
        subject_id = s["subject_id"]
        return x, y_tensor, subject_id


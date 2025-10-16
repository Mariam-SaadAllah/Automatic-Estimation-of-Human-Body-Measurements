from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

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

def build_samples(split_dir: Path) -> Tuple[list[dict[str, Any]], float, float, float, float]:
    """
    Read dataset CSV files from the given split directory (e.g., train, testA, testB)
    and build a list of sample dicts. Returns the list of samples and the min/max of 
    height and weight (for normalization reference).
    """
    # Read CSV files containing measurements, subject-to-photo mapping, and height/weight/gender.
    measurements_df = pd.read_csv(split_dir / "measurements.csv")
    subject_map_df = pd.read_csv(split_dir / "subject_to_photo_map.csv")
    hwg_meta_df = pd.read_csv(split_dir / "hwg_metadata.csv")

    # Directories containing silhouette images.
    mask_dir = split_dir / "mask"
    mask_left_dir = split_dir / "mask_left"

    # Compute min and max for height and weight from metadata (for normalization).
    h_min, h_max = hwg_meta_df["height_cm"].min(), hwg_meta_df["height_cm"].max()
    w_min, w_max = hwg_meta_df["weight_kg"].min(), hwg_meta_df["weight_kg"].max()

    # Build dictionaries for quick lookup by subject_id.
    subj_to_hw = {row["subject_id"]: row for _, row in hwg_meta_df.iterrows()}
    subj_to_measure = {row["subject_id"]: row for _, row in measurements_df.iterrows()}
    subj_to_pid = {row["subject_id"]: row["photo_id"] for _, row in subject_map_df.iterrows()}

    samples: list[dict[str, Any]] = []
    for subj, pid in subj_to_pid.items():
        frontal_img_path = mask_dir / f"{pid}.png"
        lateral_img_path = mask_left_dir / f"{pid}.png"
        if not frontal_img_path.exists() or not lateral_img_path.exists():
            # Skip this entry if images are missing.
            continue

        hw = subj_to_hw.get(subj)
        meas_row = subj_to_measure.get(subj)
        if hw is None or meas_row is None:
            # Skip if metadata or measurements are missing for this subject.
            continue

        # Raw height and weight.
        height_cm = float(hw["height_cm"])
        weight_kg = float(hw["weight_kg"])
        # Normalize height and weight to [0,1] using dataset min/max.
        height_norm = (height_cm - h_min) / (h_max - h_min) if (h_max > h_min) else 0.0
        weight_norm = (weight_kg - w_min) / (w_max - w_min) if (w_max > w_min) else 0.0

        # Collect measurement targets in cm for this subject.
        y_cm = [float(meas_row[c]) for c in MEASUREMENT_COLS]
        # Note: 'height' is included in MEASUREMENT_COLS, representing the person's height.

        samples.append({
            "photo_id": pid,
            "frontal": str(frontal_img_path),
            "lateral": str(lateral_img_path),
            "height": height_norm,
            "weight": weight_norm,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "subject_id": subj,
            "y_cm": y_cm,
            "y": y_cm,  # alias (the model will predict mm, see below)
        })

    return samples, float(h_min), float(h_max), float(w_min), float(w_max)


class BodyMDataset(Dataset):
    """
    PyTorch Dataset for BodyM silhouette data.
    Each item returns:
      - x: a tensor of shape (3, H, 2W) containing two silhouette images (frontal + lateral) 
           stacked side by side in one channel, and two extra channels for height and weight.
      - y: a tensor of shape (14,) containing the 14 target measurements in millimeters.
      - subject_id: the subject identifier.
    """
    def __init__(self, samples: list[dict[str, Any]], single_h: int = 640, single_w: int = 480, debug: bool = False):
        self.samples = samples
        self.single_h = single_h
        self.single_w = single_w
        self.debug = debug

        # ImageNet mean and std for normalization (for 3 channels).
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_resize(self, path: str) -> np.ndarray:
        """Load an image in grayscale and resize to (single_h, single_w)."""
        img = Image.open(path).convert("L")  # L mode for grayscale
        img = img.resize((self.single_w, self.single_h), resample=Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0  # scale pixel values to [0,1]
        return arr

    def __getitem__(self, idx: int):
        # Retrieve sample info
        s = self.samples[idx]
        # Load and resize both frontal and lateral silhouette images.
        front_arr = self._load_and_resize(s["frontal"])  # shape: (H, W)
        side_arr = self._load_and_resize(s["lateral"])   # shape: (H, W)
        # Concatenate images horizontally: result shape (H, 2W)
        concat_img = np.concatenate([front_arr, side_arr], axis=1).astype(np.float32)

        # Create height and weight maps of same size as concat_img.
        height_map = np.full_like(concat_img, fill_value=s["height"], dtype=np.float32)
        weight_map = np.full_like(concat_img, fill_value=s["weight"], dtype=np.float32)

        # Stack the silhouette, height, and weight into 3 channels.
        stacked = np.stack([concat_img, height_map, weight_map], axis=0)  # shape: (3, H, 2W)

        # Normalize the 3-channel image using ImageNet statistics.
        stacked = (stacked - self.imagenet_mean[:, None, None]) / self.imagenet_std[:, None, None]

        # Convert to torch.Tensor.
        x = torch.from_numpy(stacked).float()

        # Prepare target measurements in millimeters.
        y_cm = np.array(s["y_cm"], dtype=np.float32)      # 14 measurements in cm
        y_mm = torch.from_numpy(y_cm * 10.0).float()      # convert to millimeters

        subject_id = s["subject_id"]
        return x, y_mm, subject_id

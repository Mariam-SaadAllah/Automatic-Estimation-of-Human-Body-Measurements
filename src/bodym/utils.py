import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

@dataclass
class RunConfig:
    """Configuration dataclass to store training run hyperparameters and settings."""
    data_root: Path
    split: str = "train"
    batch_size: int = 22
    single_h: int = 640
    single_w: int = 480
    max_iters: int = 150_000
    reduce_iters: tuple[int, int] = (112_500, 132_000)  # iterations at 75% and 88% of max_iters
    learning_rate: float = 1e-3
    num_workers: int = 2
    out_dir: Path = Path("runs/bmnet_mnas_pretrained")
    checkpoint_dir: Path = Path("checkpoints")
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (for reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (might slow down training a bit).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(obj: dict, path: Path) -> None:
    """Save a dictionary as a JSON file to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def save_run_config(config: RunConfig, out_dir: Path) -> None:
    """Save the RunConfig dataclass as a JSON file in the output directory."""
    save_json(asdict(config), Path(out_dir) / "config.json")

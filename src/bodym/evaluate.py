from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bodym.data import BodyMDataset, build_samples
from bodym.data import Y_MIN_MM, Y_MAX_MM, MEASUREMENT_COLS
from bodym.model import MNASNetRegressor


# Helper: accuracy percentage within a tolerance (mm)

def compute_accuracy_mm(errors: np.ndarray, threshold_mm: float = 10.0) -> float:
    """
    Compute percentage of predictions within a given mm tolerance.
    Args:
        errors: numpy array of absolute errors (subjects × 14 measurements)
        threshold_mm: tolerance (default 10 mm)
    Returns:
        Accuracy percentage [0,100]
    """
    within = (errors <= threshold_mm).astype(np.float32)
    return 100.0 * within.mean()

# Subject-wise evaluation (THIS is what your professor wants for TestA/TestB)

def evaluate_subject(model: torch.nn.Module, samples: list[dict], device: str):
    """
    Subject-wise evaluation:
      1) compute per-photo error (mm)
      2) average per subject
      3) average across subjects

    Returns:
        overall_mae (float)
        tp (dict) with TP50/TP75/TP90
        per_meas_mae (np.ndarray shape (14,))
        acc_10mm (float)
    """
    model.eval()
    maes_by_subject: dict[str, list[np.ndarray]] = {}

    # constants for undo normalization: [-1,1] -> mm
    scale = torch.from_numpy(Y_MAX_MM - Y_MIN_MM).to(device)
    offset = torch.from_numpy(Y_MIN_MM).to(device)

    with torch.no_grad():
        for s in samples:
            ds = BodyMDataset([s])
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

            for x, y, sid in loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                pred_mm = (pred + 1) / 2 * scale + offset
                y_mm = (y + 1) / 2 * scale + offset

                err = torch.abs(pred_mm - y_mm).cpu().numpy()[0]  # (14,)

                subj = sid[0]
                maes_by_subject.setdefault(subj, []).append(err)

    # Average per subject -> shape (num_subjects, 14)
    subj_maes = np.array([np.mean(err_list, axis=0) for err_list in maes_by_subject.values()])

    # Per-measurement MAE and overall MAE
    per_meas_mae = subj_maes.mean(axis=0)
    overall_mae = float(per_meas_mae.mean())

    # TP metrics: measurement-wise quantile, then mean across 14 measurements
    tp = {
        "TP50": float(np.quantile(subj_maes, 0.50, axis=0).mean()),
        "TP75": float(np.quantile(subj_maes, 0.75, axis=0).mean()),
        "TP90": float(np.quantile(subj_maes, 0.90, axis=0).mean()),
    }

    # Accuracy @ 10 mm (on subject-averaged errors)
    acc_10mm = compute_accuracy_mm(subj_maes, threshold_mm=10.0)

    return overall_mae, tp, per_meas_mae, acc_10mm


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BodyM MNASNet model on test sets (subject-wise).")
    parser.add_argument("--data_root", type=Path, required=True, help="Root directory of dataset (containing testA and testB)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt or .pth file)")
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--single_h", type=int, default=640)
    parser.add_argument("--single_w", type=int, default=480)
    parser.add_argument("--splitA", type=str, default="testA", help="Name of first test split folder")
    parser.add_argument("--splitB", type=str, default="testB", help="Name of second test split folder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model and load weights
    model = MNASNetRegressor(num_outputs=14).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Build samples for test sets A and B
    testA_samples = build_samples(Path(args.data_root) / args.splitA)
    testB_samples = build_samples(Path(args.data_root) / args.splitB)

    # --- SUBJECT-WISE EVALUATION (required) ---
    overallA_mm, tpA, perA, accA = evaluate_subject(model, testA_samples, device)
    overallB_mm, tpB, perB, accB = evaluate_subject(model, testB_samples, device)

    print(f"TestA  | Overall MAE (mm): {overallA_mm:.3f}  Accuracy@10mm: {accA:.2f}%  TPs: {tpA}")
    print(f"TestB  | Overall MAE (mm): {overallB_mm:.3f}  Accuracy@10mm: {accB:.2f}%  TPs: {tpB}")

    # --- Per-measurement MAE table (required) ---
    print("\nPer-measurement MAE (mm):")
    print(f"{'Measurement':>20s} | {'TestA':>10s} | {'TestB':>10s}")
    print("-" * 45)
    for n, a, b in zip(MEASUREMENT_COLS, perA, perB):
        print(f"{n:>20s} | {a:10.2f} | {b:10.2f}")


if __name__ == "__main__":
    main()




from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bodym.data import BodyMDataset, build_samples
from bodym.data import Y_MIN_MM, Y_MAX_MM, MEASUREMENT_COLS
from bodym.model import MNASNetRegressor


def compute_accuracy_mm(errors: np.ndarray, threshold_mm: float = 10.0) -> float:
    within = (errors <= threshold_mm).astype(np.float32)
    return 100.0 * within.mean()


def evaluate_subjectwise_mm(model: torch.nn.Module, samples: list[dict], device: str):
    """
    Subject-wise evaluation:
    - average errors per subject (if multiple samples exist)
    - then average across subjects
    """
    model.eval()
    maes_by_subject: dict[str, list[np.ndarray]] = {}

    ymin = torch.from_numpy(Y_MIN_MM).to(device)
    ymax = torch.from_numpy(Y_MAX_MM).to(device)
    yrng = (ymax - ymin)

    with torch.no_grad():
        for s in samples:
            ds = BodyMDataset([s])
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            for x, y, sid in loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                pred_mm = (pred + 1) / 2 * yrng + ymin
                y_mm = (y + 1) / 2 * yrng + ymin

                err = torch.abs(pred_mm - y_mm).cpu().numpy()[0]
                maes_by_subject.setdefault(sid[0], []).append(err)

    subj_maes = np.array([np.mean(v, axis=0) for v in maes_by_subject.values()])  # (S, 14)

    per_meas_mae = subj_maes.mean(axis=0)  # (14,)
    overall_mae = float(per_meas_mae.mean())

    # FIXED: TP metrics computed per-measurement over subjects, then averaged
    tp = {
        "TP50": float(np.quantile(subj_maes, q=0.50, axis=0).mean()),
        "TP75": float(np.quantile(subj_maes, q=0.75, axis=0).mean()),
        "TP90": float(np.quantile(subj_maes, q=0.90, axis=0).mean()),
    }

    acc_10mm = compute_accuracy_mm(subj_maes, threshold_mm=10.0)

    return overall_mae, tp, per_meas_mae, acc_10mm


def evaluate_table_mm(model: torch.nn.Module, dataloader: DataLoader):
    """
    Per-sample TP distribution (not grouped by subject).
    Returns TP50/75/90 computed over all individual errors.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    device = next(model.parameters()).device

    ymin = torch.from_numpy(Y_MIN_MM).to(device)
    ymax = torch.from_numpy(Y_MAX_MM).to(device)
    yrng = (ymax - ymin)

    with torch.no_grad():
        for imgs, ys, _ in dataloader:
            imgs = imgs.to(device)
            ys = ys.to(device)
            outputs = model(imgs)

            pred_mm = (outputs + 1) / 2 * yrng + ymin
            ys_mm = (ys + 1) / 2 * yrng + ymin

            all_preds.append(pred_mm.cpu().numpy())
            all_targets.append(ys_mm.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    abs_errors = np.abs(preds - targets)  # (N, 14)

    TP50 = float(np.percentile(abs_errors, 50))
    TP75 = float(np.percentile(abs_errors, 75))
    TP90 = float(np.percentile(abs_errors, 90))
    return {"TP50": TP50, "TP75": TP75, "TP90": TP90}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BodyM MNASNet model on test sets")
    parser.add_argument("--data_root", type=Path, required=True, help="Root directory of dataset (containing testA and testB)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt or .pth file)")
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--single_h", type=int, default=640)
    parser.add_argument("--single_w", type=int, default=480)
    parser.add_argument("--splitA", type=str, default="testA", help="Name of first test split folder")
    parser.add_argument("--splitB", type=str, default="testB", help="Name of second test split folder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MNASNetRegressor(num_outputs=14).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    testA_samples = build_samples(Path(args.data_root) / args.splitA)
    testB_samples = build_samples(Path(args.data_root) / args.splitB)

    overallA_mm, tpA, perA, accA = evaluate_subjectwise_mm(model, testA_samples, device)
    overallB_mm, tpB, perB, accB = evaluate_subjectwise_mm(model, testB_samples, device)

    print(f"TestA  | Overall MAE averaged over all subjects (mm): {overallA_mm:.3f}  Accuracy@10mm: {accA:.2f}%  TPs: {tpA}")
    print(f"TestB  | Overall MAE averaged over all subjects (mm): {overallB_mm:.3f}  Accuracy@10mm: {accB:.2f}%  TPs: {tpB}")

    print("\nPer-measurement MAE (mm):")
    print(f"{'Measurement':>20s} | {'TestA':>10s} | {'TestB':>10s}")
    print("-" * 45)
    for n, a, b in zip(MEASUREMENT_COLS, perA, perB):
        print(f"{n:>20s} | {a:10.2f} | {b:10.2f}")

    testA_loader = DataLoader(
        BodyMDataset(testA_samples, single_h=args.single_h, single_w=args.single_w),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    testB_loader = DataLoader(
        BodyMDataset(testB_samples, single_h=args.single_h, single_w=args.single_w),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    metrics_testA = evaluate_table_mm(model, testA_loader)
    metrics_testB = evaluate_table_mm(model, testB_loader)

    print("\nBodyM Multi-View Evaluation Results (mm):")
    print({
        "TP90": [metrics_testA["TP90"], metrics_testB["TP90"]],
        "TP75": [metrics_testA["TP75"], metrics_testB["TP75"]],
        "TP50": [metrics_testA["TP50"], metrics_testB["TP50"]],
    })


if __name__ == "__main__":
    main()

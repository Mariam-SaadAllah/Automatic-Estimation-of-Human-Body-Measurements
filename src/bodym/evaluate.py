from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bodym.data import BodyMDataset, build_samples
from bodym.data import Y_MIN_MM, Y_MAX_MM
from bodym.model import MNASNetRegressor

def evaluate_subjectwise_mm(model: torch.nn.Module, samples: list[dict], device: str):
    """
    Evaluate model on a list of sample dicts (all from one dataset split), 
    computing MAE per subject. Returns overall MAE and a dict of TP50/75/90 percentiles.
    """
    model.eval()
    maes_by_subject: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for s in samples:
            ds = BodyMDataset([s])
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            for x, y, sid in loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                # Unnormalize predictions and ground-truths back to millimeters
                pred_mm = (pred + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)
                y_mm = (y + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)

                # Compute absolute errors in millimeters
                err = torch.abs(pred_mm - y_mm).cpu().numpy()[0]
                maes_by_subject.setdefault(sid[0], []).append(err)
    # Compute mean error per subject (average across multiple images of same subject if any)
    subj_maes = np.array([np.mean(v, axis=0) for v in maes_by_subject.values()])
    overall_mae = float(subj_maes.mean())  # mean of all errors
    # Compute percentile thresholds (TP50, TP75, TP90) over all absolute errors
    tp = {
        "TP50": float(np.percentile(np.abs(subj_maes), 50)),
        "TP75": float(np.percentile(np.abs(subj_maes), 75)),
        "TP90": float(np.percentile(np.abs(subj_maes), 90)),
    }
    return overall_mae, tp

def evaluate_table_mm(model: torch.nn.Module, dataloader: DataLoader):
    """
    Evaluate model on a dataloader (one full dataset) and compute overall TP50, TP75, TP90 
    on a per-sample basis. (This treats each prediction independently, not grouped by subject.)
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for imgs, ys, _ in dataloader:
            imgs = imgs.to(device)
            ys = ys.to(device)
            outputs = model(imgs)
            # Unnormalize both outputs and targets to millimeters
            pred_mm = (outputs + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)
            ys_mm = (ys + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)

            all_preds.append(pred_mm.cpu().numpy())
            all_targets.append(ys_mm.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    abs_errors = np.abs(preds - targets)  # shape: (N, 14)
    # Compute distribution percentiles over all individual errors
    TP50 = np.percentile(abs_errors, 50)
    TP75 = np.percentile(abs_errors, 75)
    TP90 = np.percentile(abs_errors, 90)
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

    # Initialize model and load weights
    model = MNASNetRegressor(num_outputs=14)
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])  # if loading from full checkpoint
    else:
        model.load_state_dict(checkpoint)  # if loading state dict directly

    # Build samples for test sets A and B
    testA_samples = build_samples(Path(args.data_root) / args.splitA)
    testB_samples = build_samples(Path(args.data_root) / args.splitB)


    # Evaluate subject-wise (each subject's mean error)
    overallA_mm, tpA = evaluate_subjectwise_mm(model, testA_samples, device)
    overallB_mm, tpB = evaluate_subjectwise_mm(model, testB_samples, device)
    print(f"TestA overall MAE (mm): {overallA_mm:.3f} TPs: {tpA}")
    print(f"TestB overall MAE (mm): {overallB_mm:.3f} TPs: {tpB}")

    # Evaluate on entire set with batched dataloader (for overall error distribution)
    testA_loader = DataLoader(BodyMDataset(testA_samples, single_h=args.single_h, single_w=args.single_w),
                              batch_size=args.batch_size, shuffle=False, num_workers=2)
    testB_loader = DataLoader(BodyMDataset(testB_samples, single_h=args.single_h, single_w=args.single_w),
                              batch_size=args.batch_size, shuffle=False, num_workers=2)
    metrics_testA = evaluate_table_mm(model, testA_loader)
    metrics_testB = evaluate_table_mm(model, testB_loader)

    print("\nBodyM Multi-View Evaluation Results (mm)")
    print({
        "TP90": [metrics_testA["TP90"], metrics_testB["TP90"]],
        "TP75": [metrics_testA["TP75"], metrics_testB["TP75"]],
        "TP50": [metrics_testA["TP50"], metrics_testB["TP50"]],
    })

if __name__ == "__main__":
    main()




from __future__ import annotations

import argparse
import math
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bodym.data import BodyMDataset, build_samples
from bodym.data import Y_MIN_MM, Y_MAX_MM
from bodym.data import MEASUREMENT_COLS
from bodym.model import MNASNetRegressor
from bodym.utils import RunConfig, seed_everything, save_run_config


# Accuracy helper

def compute_accuracy_mm(errors: np.ndarray, threshold_mm: float = 10.0) -> float:
    within = (errors <= threshold_mm).astype(np.float32)
    return 100.0 * within.mean()


def reduce_lr_by_factor(opt: torch.optim.Optimizer, factor: float = 0.1) -> None:
    for pg in opt.param_groups:
        pg["lr"] *= factor


# subject evaluation (for TestA / TestB only)

def evaluate_subject(
    model: nn.Module,
    sample_list: list[dict],
    device: str,
) -> tuple[np.ndarray, float, dict[str, float], float]:
    """
    Subject-wise evaluation:
    - average errors per subject
    - then average across subjects
    """
    model.eval()
    maes_by_subject: dict[str, list[np.ndarray]] = {}

    ymin = torch.from_numpy(Y_MIN_MM).to(device)
    ymax = torch.from_numpy(Y_MAX_MM).to(device)
    yrng = (ymax - ymin)

    with torch.no_grad():
        for s in sample_list:
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

    subj_maes = np.array([np.mean(v, axis=0) for v in maes_by_subject.values()])
    per_measurement_mae = subj_maes.mean(axis=0)
    overall_mae = float(per_measurement_mae.mean())

    tp = {
        "TP50": float(np.quantile(subj_maes, q=0.50, axis=0).mean()),
        "TP75": float(np.quantile(subj_maes, q=0.75, axis=0).mean()),
        "TP90": float(np.quantile(subj_maes, q=0.90, axis=0).mean()),
    }

    acc_10mm = compute_accuracy_mm(subj_maes, threshold_mm=10.0)
    return per_measurement_mae, overall_mae, tp, acc_10mm



# sample evaluation (for training / validation)

def evaluate_sample(
    model: nn.Module,
    sample_list: list[dict],
    device: str,
    single_h: int,
    single_w: int,
    batch_size: int,
    num_workers: int,
):
    model.eval()
    ds = BodyMDataset(sample_list, single_h=single_h, single_w=single_w)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ymin = torch.from_numpy(Y_MIN_MM).to(device)
    ymax = torch.from_numpy(Y_MAX_MM).to(device)
    yrng = (ymax - ymin)

    all_errs = []
    loss_vals = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss_vals.append(torch.abs(pred - y).mean().item())

            pred_mm = (pred + 1) / 2 * yrng + ymin
            y_mm = (y + 1) / 2 * yrng + ymin

            err_mm = torch.abs(pred_mm - y_mm).cpu().numpy()
            all_errs.append(err_mm)

    errs = np.concatenate(all_errs, axis=0)
    per_meas_mae = errs.mean(axis=0)
    overall_mae = float(per_meas_mae.mean())

    tp = {
        "TP50": float(np.quantile(errs, 0.50, axis=0).mean()),
        "TP75": float(np.quantile(errs, 0.75, axis=0).mean()),
        "TP90": float(np.quantile(errs, 0.90, axis=0).mean()),
    }

    acc_10mm = compute_accuracy_mm(errs, threshold_mm=10.0)
    loss_norm = float(np.mean(loss_vals))

    return per_meas_mae, overall_mae, tp, acc_10mm, loss_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--single_h", type=int, default=640)
    parser.add_argument("--single_w", type=int, default=480)
    parser.add_argument("--max_iters", type=int, default=150_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=Path, default=Path("runs/bmnet"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    samples = build_samples(Path(args.data_root) / args.split)

    #  split samples 10% for validation (5492 / 642)
    train_samples, val_samples = train_test_split(
        samples, test_size=642, random_state=args.seed, shuffle=True
    )

    train_ds = BodyMDataset(train_samples, single_h=args.single_h, single_w=args.single_w)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = MNASNetRegressor(num_outputs=14).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    writer = SummaryWriter(log_dir=str(args.out_dir))
    scaler = GradScaler(enabled=(device == "cuda"))

    iteration = 0
    best_val = float("inf")

    for epoch in range(1000):
        model.train()
        running_loss = 0.0

        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast(enabled=(device == "cuda")):
                loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            iteration += 1
            if iteration >= args.max_iters:
                break

        avg_train_loss = running_loss / len(loader)
        writer.add_scalar("train/loss", avg_train_loss, epoch)

        per, overall, tp, acc, val_loss = evaluate_sample(
            model, val_samples, device,
            args.single_h, args.single_w,
            args.batch_size, args.num_workers
        )

        writer.add_scalar("val/loss", val_loss, epoch)

        if (epoch + 1) % 5 == 0:
            print("\nPer-measurement MAE (mm):")
            for n, v in zip(MEASUREMENT_COLS, per):
                print(f"{n:>20s}: {v:7.2f} mm")

        if overall < best_val:
            best_val = overall
            torch.save(model.state_dict(), args.checkpoint_dir / "best.pt")

        if iteration >= args.max_iters:
            break


if __name__ == "__main__":
    main()


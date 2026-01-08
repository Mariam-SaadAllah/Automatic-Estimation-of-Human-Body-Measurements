from __future__ import annotations

import argparse
import math
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
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


# Helper functions
def compute_accuracy_mm(errors: np.ndarray, threshold_mm: float = 10.0) -> float:
    within = (errors <= threshold_mm).astype(np.float32)
    return 100.0 * within.mean()


def reduce_lr_by_factor(opt: torch.optim.Optimizer, factor: float = 0.1) -> None:
    for pg in opt.param_groups:
        pg["lr"] *= factor


def evaluate_subjectwise_model(model: nn.Module, sample_list: list[dict], device: str):
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

    subj_maes = np.array([np.mean(errors, axis=0) for errors in maes_by_subject.values()])
    per_measurement_mae = subj_maes.mean(axis=0)
    overall_mae = float(per_measurement_mae.mean())
    tp = {
        "TP50": float(np.quantile(subj_maes, q=0.50, axis=0).mean()),
        "TP75": float(np.quantile(subj_maes, q=0.75, axis=0).mean()),
        "TP90": float(np.quantile(subj_maes, q=0.90, axis=0).mean()),
    }
    accuracy_10mm = compute_accuracy_mm(subj_maes, threshold_mm=10.0)
    return per_measurement_mae, overall_mae, tp, accuracy_10mm


# ✅ NEW: compute validation loss in normalized space (same as training loss)
def evaluate_val_loss_norm(
    model: nn.Module,
    sample_list: list[dict],
    device: str,
    single_h: int,
    single_w: int,
    batch_size: int,
    num_workers: int,
) -> float:
    model.eval()
    ds = BodyMDataset(sample_list, single_h=single_h, single_w=single_w)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.L1Loss(reduction="mean")
    losses = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(float(criterion(pred, y).item()))

    return float(np.mean(losses)) if len(losses) > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BodyM MNASNet Regressor model.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--single_h", type=int, default=640)
    parser.add_argument("--single_w", type=int, default=480)
    parser.add_argument("--max_iters", type=int, default=150_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)

    DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/BMNet_Project")
    DEFAULT_OUT_DIR = DEFAULT_DRIVE_ROOT / "runs/fine_tune_full"
    DEFAULT_CKPT_DIR = DEFAULT_DRIVE_ROOT / "checkpoints/fine_tune_full"
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CKPT_DIR)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1",
                        choices=["IMAGENET1K_V1", "DEFAULT", "NONE"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    split_dir = Path(args.data_root) / args.split
    samples = build_samples(split_dir)

    # ---------------- SAMPLE-WISE 90/10 SPLIT ----------------
    if args.split == "train":
        train_samples, val_samples = train_test_split(
            samples, test_size=0.1, random_state=args.seed, shuffle=True
        )
        print(f"Training samples:  {len(train_samples)}, Validation samples:  {len(val_samples)}")
    else:
        train_samples = samples
        val_samples = samples

    dataset = BodyMDataset(train_samples, single_h=args.single_h, single_w=args.single_w)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = MNASNetRegressor(num_outputs=14, weights=None if args.weights == "NONE" else args.weights)
    model = model.to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Backbone layers frozen — only regression head will be trained.")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.L1Loss()
    writer = SummaryWriter(log_dir=str(args.out_dir))

    train_size = len(loader.dataset)
    iters_per_epoch = math.ceil(train_size / args.batch_size)
    max_iters = args.max_iters
    reduce_iters = [int(0.75 * max_iters), int(0.88 * max_iters)]
    num_epochs = math.ceil(max_iters / iters_per_epoch)

    save_run_config(RunConfig(
        data_root=Path(args.data_root),
        split=args.split,
        batch_size=args.batch_size,
        single_h=args.single_h,
        single_w=args.single_w,
        max_iters=args.max_iters,
        reduce_iters=tuple(reduce_iters),
        learning_rate=args.lr,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device=device,
    ), args.out_dir)

    scaler = GradScaler(enabled=(device == "cuda"))

    # ---------------- RESUME ----------------
    args.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    latest_ckpt = args.checkpoint_dir / "latest_checkpoint.pth"

    start_epoch = 0
    iteration = 0
    best_val = float("inf")

    if latest_ckpt.exists():
        print(f"Resuming training from: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", scaler.state_dict()))
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        start_epoch = checkpoint.get("epoch", 0)
        iteration = checkpoint.get("iteration", 0)
        best_val = checkpoint.get("best_val", float("inf"))
    else:
        print("Starting training from scratch...")

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for x_batch, y_batch, _ in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                preds = model(x_batch)
                loss = criterion(preds, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            batch_count += 1
            iteration += 1

            if iteration in reduce_iters:
                reduce_lr_by_factor(optimizer, factor=0.1)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)

            if iteration >= max_iters:
                break

        avg_train_loss = running_loss / max(1, batch_count)

        val_subset = val_samples if len(val_samples) <= 1000 else list(
            np.random.default_rng(args.seed).choice(val_samples, size=min(256, len(val_samples)), replace=False)
        )

        # subject-wise mm metrics (as before)
        per_meas_mae, overall_mae, tp, acc_10mm = evaluate_subjectwise_model(model, val_subset, device)

        # normalized validation loss (same scale as training loss)
        val_loss_norm = evaluate_val_loss_norm(
            model=model,
            sample_list=val_subset,
            device=device,
            single_h=args.single_h,
            single_w=args.single_w,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("val/loss_epoch", val_loss_norm, epoch)          #val loss
        writer.add_scalar("val/overall_mae_mm", overall_mae, epoch)
        writer.add_scalar("val/accuracy_10mm", acc_10mm, epoch)

        print(
            f"Epoch {epoch+1}/{num_epochs}  Iter {iteration}  "
            f"Train loss: {avg_train_loss:.6f}  "
            f"Val loss: {val_loss_norm:.6f}  "
            f"Val overall MAE (mm): {overall_mae:.3f}"
        )

        # ======= PRINT 14 MEASUREMENTS EVERY 4 EPOCHS =======
        if (epoch + 1) % 4 == 0:
            print("\nPer-measurement MAE (mm) (every 4 epochs):")
            for name, mae in zip(MEASUREMENT_COLS, per_meas_mae):
                print(f"  {name:>20s}: {mae:7.2f} mm")
            print()

        # ======= SAVE CHECKPOINTS =======
        ckpt = {
            "epoch": epoch + 1,
            "iteration": iteration,
            "loss": avg_train_loss,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        torch.save(ckpt, args.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(ckpt, args.checkpoint_dir / "latest_checkpoint.pth")

        if overall_mae < best_val:
            best_val = overall_mae
            torch.save(ckpt, args.checkpoint_dir / "best_checkpoint.pth")
            torch.save(model.state_dict(), args.checkpoint_dir / "best_mnasnet_bmnet.pt")

        if iteration >= max_iters:
            break

    print("Training complete. Best validation overall MAE (mm):", best_val)


if __name__ == "__main__":
    main()


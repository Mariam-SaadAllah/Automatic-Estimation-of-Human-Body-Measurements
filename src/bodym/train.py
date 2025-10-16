from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bodym.data import BodyMDataset, build_samples
from bodym.model import MNASNetRegressor
from bodym.utils import RunConfig, seed_everything, save_run_config

def reduce_lr_by_factor(opt: torch.optim.Optimizer, factor: float = 0.1) -> None:
    """Scale down the learning rate of each parameter group by the given factor."""
    for pg in opt.param_groups:
        pg["lr"] *= factor

def evaluate_subjectwise_model(model: nn.Module, sample_list: list[dict], device: str) -> tuple[np.ndarray, float, dict[str, float]]:
    """
    Evaluate the model on a list of samples (dictionaries) and compute MAE per measurement for each subject.
    Returns (per_measurement_mae, overall_mae, tp_dict) where tp_dict contains mean TP50, TP75, TP90 across measurements.
    """
    model.eval()
    maes_by_subject: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for s in sample_list:
            # Create a dataset for this single sample (subject).
            ds = BodyMDataset([s])
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            for x, y, sid in loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                err = torch.abs(pred - y).cpu().numpy()[0]  # absolute error for this sample (numpy array of shape 14)
                maes_by_subject.setdefault(sid[0], []).append(err)
    # Compute mean error per measurement for each subject, then average across subjects.
    subj_maes = np.array([np.mean(errors, axis=0) for errors in maes_by_subject.values()])  # shape: (num_subjects, 14)
    per_measurement_mae = subj_maes.mean(axis=0)  # average MAE for each of the 14 measurements
    overall_mae = float(per_measurement_mae.mean())  # average MAE across all measurements
    # Compute percentile thresholds (TP50/75/90) across all subject-wise errors, averaged over measurements.
    tp = {}
    tp["TP50"] = float(np.quantile(subj_maes, q=0.50, axis=0).mean())
    tp["TP75"] = float(np.quantile(subj_maes, q=0.75, axis=0).mean())
    tp["TP90"] = float(np.quantile(subj_maes, q=0.90, axis=0).mean())
    return per_measurement_mae, overall_mae, tp

def main() -> None:
    parser = argparse.ArgumentParser(description="Train BodyM MNASNet Regressor model.")
    parser.add_argument("--data_root", type=Path, required=True, help="Root directory containing train/testA/testB subfolders")
    parser.add_argument("--split", type=str, default="train", help="Which split folder to use for training (default: train)")
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--single_h", type=int, default=640, help="Height to resize each silhouette image")
    parser.add_argument("--single_w", type=int, default=480, help="Width to resize each silhouette image")
    parser.add_argument("--max_iters", type=int, default=150_000, help="Total training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers for loading images")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/bmnet_mnas_pretrained"), help="Directory to save logs and outputs")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"), help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1", choices=["IMAGENET1K_V1", "DEFAULT", "NONE"],
                        help="Pre-trained weights for MNASNet (use 'NONE' for random init)")
    parser.add_argument("--eval_only", action="store_true", help="If set, perform only a quick evaluation on the split and exit")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    # Prepare dataset samples
    split_dir = Path(args.data_root) / args.split
    samples, h_min, h_max, w_min, w_max = build_samples(split_dir)

    dataset = BodyMDataset(samples, single_h=args.single_h, single_w=args.single_w)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize model and optimizer
    model = MNASNetRegressor(num_outputs=14, weights=None if args.weights == "NONE" else args.weights)
    model = model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.L1Loss()  # Mean Absolute Error loss (in millimeters)

    # Prepare TensorBoard writer and training schedule
    writer = SummaryWriter(log_dir=str(args.out_dir))
    train_size = len(loader.dataset)
    iters_per_epoch = math.ceil(train_size / args.batch_size)
    max_iters = args.max_iters
    # Define iterations at which to reduce LR (75% and 88% of training)
    reduce_iters = [int(0.75 * max_iters), int(0.88 * max_iters)]
    num_epochs = math.ceil(max_iters / iters_per_epoch)

    # Save run configuration for reference
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

    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=(device == "cuda"))

    iteration = 0
    best_val = float("inf")
    args.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    if args.eval_only:
        # If eval_only, evaluate on the provided split and exit.
        _, overall_mae, tp = evaluate_subjectwise_model(model, samples, device)
        print(f"Eval-only: overall MAE (mm): {overall_mae:.3f}  TPs: {tp}")
        return

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for x_batch, y_batch, _ in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            # Forward pass with automatic mixed precision
            with autocast(enabled=(device == "cuda")):
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
            # Backpropagate scaled loss and update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            batch_count += 1
            iteration += 1

            # Adjust learning rate at scheduled milestones
            if iteration in reduce_iters:
                reduce_lr_by_factor(optimizer, factor=0.1)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)

            if iteration >= max_iters:
                break  # exit early if reached max iterations

        avg_train_loss = running_loss / max(1, batch_count)

        # Quick validation on a subset (or full set if small) of training data to estimate performance
        if len(samples) > 1000:
            rng = np.random.default_rng(args.seed)
            val_subset = list(rng.choice(samples, size=min(256, len(samples)), replace=False))
        else:
            val_subset = samples
        _, overall_mae, tp = evaluate_subjectwise_model(model, val_subset, device)
        writer.add_scalar("val/overall_mae_mm", overall_mae, epoch)
        writer.add_text("val/TP", str(tp), epoch)
        print(f"Epoch {epoch+1}/{num_epochs}  Iter {iteration}  Train loss: {avg_train_loss:.6f}  "
              f"Val overall MAE (mm): {overall_mae:.3f}  TPs: {tp}")

        # Save checkpoint for this epoch
        ckpt = {
            "epoch": epoch + 1,
            "iteration": iteration,
            "loss": avg_train_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(ckpt, args.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

        # If best validation performance, save the model weights separately
        if overall_mae < best_val:
            best_val = overall_mae
            torch.save(model.state_dict(), args.checkpoint_dir / "best_mnasnet_bmnet.pt")

        if iteration >= max_iters:
            break  # end training if max iterations reached

    print("Training complete. Best validation overall MAE (mm):", best_val)


if __name__ == "__main__":
    main()

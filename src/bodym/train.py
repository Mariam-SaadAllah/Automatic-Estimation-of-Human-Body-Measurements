from __future__ import annotations

import argparse
import math
from pathlib import Path
import collections
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bodym.data import BodyMDataset, build_samples
from bodym.data import Y_MIN_MM, Y_MAX_MM
from bodym.model import MNASNetRegressor
from bodym.utils import RunConfig, seed_everything, save_run_config

# ---------------------------------------------------------------------
# NEW: Helper function for accuracy
# ---------------------------------------------------------------------
def compute_accuracy_mm(errors: np.ndarray, threshold_mm: float = 10.0) -> float:
    """
    Compute percentage of measurements whose absolute error is <= threshold_mm.
    Args:
        errors: numpy array of absolute errors (subjects × 14)
        threshold_mm: tolerance in millimeters
    Returns:
        Accuracy percentage [0,100]
    """
    within = (errors <= threshold_mm).astype(np.float32)
    return 100.0 * within.mean()
# ---------------------------------------------------------------------

def reduce_lr_by_factor(opt: torch.optim.Optimizer, factor: float = 0.1) -> None:
    """Scale down the learning rate of each parameter group by the given factor."""
    for pg in opt.param_groups:
        pg["lr"] *= factor


# ---------------------------------------------------------------------
# UPDATED: Now returns accuracy_10mm as well
# ---------------------------------------------------------------------
def evaluate_subjectwise_model(model: nn.Module, sample_list: list[dict], device: str) -> tuple[np.ndarray, float, dict[str, float], float]:
    """
    Evaluate the model on a list of samples (dictionaries) and compute MAE per measurement for each subject.
    Returns (per_measurement_mae, overall_mae, tp_dict, accuracy_10mm)
    """
    model.eval()
    maes_by_subject: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for s in sample_list:
            ds = BodyMDataset([s])
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            for x, y, sid in loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                # Unnormalize model outputs and targets back to millimeters
                pred_mm = (pred + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)
                y_mm = (y + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)

                # Compute absolute error in millimeters
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

    # NEW: compute accuracy within ±10 mm
    accuracy_10mm = compute_accuracy_mm(subj_maes, threshold_mm=10.0)

    return per_measurement_mae, overall_mae, tp, accuracy_10mm
# ---------------------------------------------------------------------


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
    parser.add_argument("--freeze_backbone", action="store_true", help="If set, freeze the MNASNet feature extractor layers")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    # Prepare dataset samples
    split_dir = Path(args.data_root) / args.split
    samples = build_samples(split_dir)

    # ---------------- SUBJECT-BASED 90/10 TRAIN-VALIDATION SPLIT ----------------
    if args.split == "train":
        subject_to_samples = collections.defaultdict(list)
        for s in samples:
            subject_to_samples[s["subject_id"]].append(s)
        subject_ids = list(subject_to_samples.keys())

        train_subjects, val_subjects = train_test_split(
            subject_ids, test_size=0.1, random_state=args.seed
        )

        train_samples = [s for sid in train_subjects for s in subject_to_samples[sid]]
        val_samples = [s for sid in val_subjects for s in subject_to_samples[sid]]

        print(f"Training subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}")
        print(f"Training samples:  {len(train_samples)}, Validation samples:  {len(val_samples)}")
    else:
        train_samples = samples
        val_samples = samples
    # ---------------------------------------------------------------------------

    dataset = BodyMDataset(train_samples, single_h=args.single_h, single_w=args.single_w)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize model and optimizer
    model = MNASNetRegressor(num_outputs=14, weights=None if args.weights == "NONE" else args.weights)
    model = model.to(device)

    # ---------------- FREEZE BACKBONE IF REQUESTED ----------------
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:  # keep final regression head trainable
                param.requires_grad = False
        print("Backbone layers frozen — only regression head will be trained.")
    # ---------------------------------------------------------------

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.L1Loss()  # Mean Absolute Error loss (in millimeters)

    # Prepare TensorBoard writer and training schedule
    writer = SummaryWriter(log_dir=str(args.out_dir))
    train_size = len(loader.dataset)
    iters_per_epoch = math.ceil(train_size / args.batch_size)
    max_iters = args.max_iters
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
        _, overall_mae, tp, acc_10mm = evaluate_subjectwise_model(model, val_samples, device)
        print(f"Eval-only: overall MAE (mm): {overall_mae:.3f}  Accuracy@10mm: {acc_10mm:.2f}%  TPs: {tp}")
        return

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(num_epochs):
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

        # ---------------- VALIDATION ----------------
        if args.split == "train":
            val_subset = val_samples if len(val_samples) <= 1000 else list(
                np.random.default_rng(args.seed).choice(val_samples, size=min(256, len(val_samples)), replace=False)
            )
        else:
            val_subset = val_samples

        # UPDATED VALIDATION BLOCK
        per_meas_mae, overall_mae, tp, acc_10mm = evaluate_subjectwise_model(model, val_subset, device)

        # --- Log metrics to TensorBoard ---
        writer.add_scalar("val/overall_mae_mm", overall_mae, epoch)
        writer.add_scalar("val/accuracy_10mm", acc_10mm, epoch)
        writer.add_scalar("val/TP50_mm", tp["TP50"], epoch)
        writer.add_scalar("val/TP75_mm", tp["TP75"], epoch)
        writer.add_scalar("val/TP90_mm", tp["TP90"], epoch)
        writer.add_text("val/TP", str(tp), epoch)

        # --- Optional per-measurement logging ---
        from bodym.data import MEASUREMENT_COLS
        for name, mae in zip(MEASUREMENT_COLS, per_meas_mae):
            writer.add_scalar(f"val/mae_{name}_mm", mae, epoch)

        # --- Console summary ---
        print(f"Epoch {epoch+1}/{num_epochs}  Iter {iteration}  "
              f"Train loss: {avg_train_loss:.6f}  "
              f"Val overall MAE (mm): {overall_mae:.3f}  "
              f"Accuracy@10mm: {acc_10mm:.2f}%  "
              f"TPs: {tp}")

        # Print per-epoch summary line (always)
        print(f"Epoch {epoch+1}/{num_epochs}  Iter {iteration}  Train loss: {avg_train_loss:.6f}  "
              f"Val overall MAE (mm): {overall_mae:.3f}  Accuracy@10mm: {acc_10mm:.2f}%  TPs: {tp}")

        # Print the full 14-measurement MAE table every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\nFull per-measurement MAE (mm):")
            for name, mae in zip(MEASUREMENT_COLS, per_meas_mae):
                print(f"  {name:>20s}: {mae:7.2f} mm")
            print()  # blank line for readability

        ckpt = {
            "epoch": epoch + 1,
            "iteration": iteration,
            "loss": avg_train_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(ckpt, args.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

        if overall_mae < best_val:
            best_val = overall_mae
            torch.save(model.state_dict(), args.checkpoint_dir / "best_mnasnet_bmnet.pt")

        if iteration >= max_iters:
            break

    print("Training complete. Best validation overall MAE (mm):", best_val)


if __name__ == "__main__":
    main()


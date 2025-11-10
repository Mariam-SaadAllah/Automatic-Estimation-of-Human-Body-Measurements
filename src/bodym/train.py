from __future__ import annotations

import argparse
import math
from pathlib import Path
import collections
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


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def compute_accuracy_mm(errors: np.ndarray, threshold_mm: float = 10.0) -> float:
    within = (errors <= threshold_mm).astype(np.float32)
    return 100.0 * within.mean()


def reduce_lr_by_factor(opt: torch.optim.Optimizer, factor: float = 0.1) -> None:
    for pg in opt.param_groups:
        pg["lr"] *= factor
# ---------------------------------------------------------------------


def evaluate_subjectwise_model(model: nn.Module, sample_list: list[dict], device: str):
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
                pred_mm = (pred + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)
                y_mm = (y + 1) / 2 * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)
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
    # ======= COLAB DRIVE PATHS =======
    DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/BMNet_Project")
    DEFAULT_OUT_DIR = DEFAULT_DRIVE_ROOT / "runs/fine_tune_full"
    DEFAULT_CKPT_DIR = DEFAULT_DRIVE_ROOT / "checkpoints/fine_tune_full"
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CKPT_DIR)
    # ================================
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

    if args.split == "train":
        subject_to_samples = collections.defaultdict(list)
        for s in samples:
            subject_to_samples[s["subject_id"]].append(s)
        subject_ids = list(subject_to_samples.keys())
        train_subjects, val_subjects = train_test_split(subject_ids, test_size=0.1, random_state=args.seed)
        train_samples = [s for sid in train_subjects for s in subject_to_samples[sid]]
        val_samples = [s for sid in val_subjects for s in subject_to_samples[sid]]
        print(f"Training subjects: {len(train_subjects)}, Validation subjects: {len(val_subjects)}")
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
    
    # --- Model parameter statistics ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")


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

    # --- Resume training if a checkpoint exists ---
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
        torch.set_rng_state(checkpoint.get("rng_state", torch.get_rng_state()))
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_epoch = checkpoint.get("epoch", 0)
        iteration = checkpoint.get("iteration", 0)
        best_val = checkpoint.get("best_val", float("inf"))
        print(f"Resumed at epoch {start_epoch}, iteration {iteration}, best_val {best_val:.3f}")
    else:
        print("Starting training from scratch...")


    if args.eval_only:
        best_ckpt = args.checkpoint_dir / "best_checkpoint.pth"
        latest_ckpt = args.checkpoint_dir / "latest_checkpoint.pth"

        # Prefer best checkpoint, fallback to latest if best not found
        if best_ckpt.exists():
            print(f"Evaluating from best checkpoint: {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device)
        elif latest_ckpt.exists():
            print(f"Warning: best_checkpoint.pth not found, evaluating from latest checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
        else:
            raise FileNotFoundError("No checkpoint found for evaluation.")

        model.load_state_dict(checkpoint["model_state_dict"])

        # Run evaluation
        _, overall_mae, tp, acc_10mm = evaluate_subjectwise_model(model, val_samples, device)
        print("\n==================== EVALUATION RESULTS ====================")
        print(f"Overall MAE (mm): {overall_mae:.3f}")
        print(f"Accuracy ≤10 mm:  {acc_10mm:.2f}%")
        print(f"TPs: {tp}")
        print("============================================================\n")
        return

    metrics_log = []

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

        if args.split == "train":
            val_subset = val_samples if len(val_samples) <= 1000 else list(
                np.random.default_rng(args.seed).choice(val_samples, size=min(256, len(val_samples)), replace=False)
            )
        else:
            val_subset = val_samples

        per_meas_mae, overall_mae, tp, acc_10mm = evaluate_subjectwise_model(model, val_subset, device)

        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("val/overall_mae_mm", overall_mae, epoch)
        writer.add_scalar("val/accuracy_10mm", acc_10mm, epoch)
        writer.add_scalar("val/TP50_mm", tp["TP50"], epoch)
        writer.add_scalar("val/TP75_mm", tp["TP75"], epoch)
        writer.add_scalar("val/TP90_mm", tp["TP90"], epoch)

        for name, mae in zip(MEASUREMENT_COLS, per_meas_mae):
            writer.add_scalar(f"val/mae_{name}_mm", mae, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}  Iter {iteration}  Train loss: {avg_train_loss:.6f}  "
              f"Val overall MAE (mm): {overall_mae:.3f}  Accuracy@10mm: {acc_10mm:.2f}%  TPs: {tp}")

        if (epoch + 1) % 5 == 0:
            print("\nFull per-measurement MAE (mm):")
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
            "scaler_state_dict": scaler.state_dict(),  # NEW: for mixed precision
            "rng_state": torch.get_rng_state(),        # NEW: for reproducibility
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(ckpt, args.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(ckpt, args.checkpoint_dir / "latest_checkpoint.pth")

        if overall_mae < best_val:
            best_val = overall_mae
            torch.save(ckpt, args.checkpoint_dir / "best_checkpoint.pth")
            torch.save(model.state_dict(), args.checkpoint_dir / "best_mnasnet_bmnet.pt")
        # =================================

        metrics_log.append({
            "epoch": epoch + 1,
            "iteration": iteration,
            "train_loss": avg_train_loss,
            "val_overall_mae_mm": overall_mae,
            "val_accuracy_10mm": acc_10mm,
            "TP50_mm": tp["TP50"],
            "TP75_mm": tp["TP75"],
            "TP90_mm": tp["TP90"],
            **{f"mae_{name}_mm": mae for name, mae in zip(MEASUREMENT_COLS, per_meas_mae)}
        })

        if iteration >= max_iters:
            break

    df = pd.DataFrame(metrics_log)
    csv_path = args.out_dir / "epoch_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed epoch metrics to {csv_path}")

    with open(args.out_dir / "training_summary.txt", "w") as f:
        f.write(f"Best Validation MAE (mm): {best_val:.3f}\n")
        f.write(f"Total epochs: {epoch+1}\n")
        f.write(f"Final iteration: {iteration}\n")
        f.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
        f.write("Per-measurement MAE (mm):\n")
        for name, mae in zip(MEASUREMENT_COLS, per_meas_mae):
            f.write(f"  {name:>20s}: {mae:7.2f}\n")
    print("Saved training summary for thesis documentation.")
    print("Training complete. Best validation overall MAE (mm):", best_val)


if __name__ == "__main__":
    main()



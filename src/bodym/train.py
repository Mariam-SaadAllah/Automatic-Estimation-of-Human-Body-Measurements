import argparse
import os
import random
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# === Import local modules (from your project) ===
from bodym.data import BodyMDataset, build_samples
from bodym.model import MNASNetRegressor
from bodym.utils import RunConfig, seed_everything, save_run_config


# -------------------- Utility functions --------------------
def reduce_lr_by_factor(optimizer: torch.optim.Optimizer, factor: float = 0.1):
    """Multiply LR of every param group by `factor`."""
    new_lrs = []
    for pg in optimizer.param_groups:
        pg["lr"] = pg["lr"] * factor
        new_lrs.append(pg["lr"])
    return new_lrs


def group_params(model: nn.Module,
                 head_keywords: Tuple[str, ...] = ("regressor", "head", "classifier")):
    """Return backbone vs head params by matching parameter names."""
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in head_keywords):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return backbone_params, head_params


@torch.no_grad()
def evaluate_subjectwise_model(
    model: nn.Module,
    sample_list: List[Dict],
    device: torch.device,
    subject_map: Optional[Dict[str, str]] = None,
    batch_size: int = 1,
    num_workers: int = 2,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Evaluate model MAE per subject, overall MAE, and TP50/75/90."""
    model.eval()
    subj_errs: Dict[str, List[np.ndarray]] = {}

    eval_ds = BodyMDataset(sample_list)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for x, y, sid, meta in eval_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        abs_err = (preds - y).abs().detach().cpu().numpy()  # shape [B, 14]

        sids = sid
        if subject_map is not None and "photo_id" in meta:
            photo_ids = meta["photo_id"]
            mapped = [subject_map.get(pid, sid_i) for pid, sid_i in zip(photo_ids, sids)]
            sids = mapped

        for err_vec, subject_id in zip(abs_err, sids):
            subj_errs.setdefault(str(subject_id), []).append(err_vec)

    subj_maes = [np.mean(np.stack(v, axis=0), axis=0) for v in subj_errs.values()]
    subj_maes = np.stack(subj_maes, axis=0)  # [num_subjects, 14]

    per_measurement_mae = np.mean(subj_maes, axis=0)
    overall_mae = float(np.mean(per_measurement_mae))

    tp50 = float(np.mean(np.quantile(subj_maes, 0.50, axis=0)))
    tp75 = float(np.mean(np.quantile(subj_maes, 0.75, axis=0)))
    tp90 = float(np.mean(np.quantile(subj_maes, 0.90, axis=0)))
    tp = {"TP50": tp50, "TP75": tp75, "TP90": tp90}

    return per_measurement_mae, overall_mae, tp


def read_subject_map_csv(csv_path: Path) -> Dict[str, str]:
    """Read subject_to_photo_map.csv and return {photo_id -> subject_id} dict."""
    import csv
    mapping = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = {k.lower(): k for k in row.keys()}
            subj_key = next((keys[k] for k in keys if k in ("subject_id", "subject", "sid")), None)
            photo_key = next((keys[k] for k in keys if k in ("photo_id", "photo", "pid", "image_id")), None)
            if subj_key is None or photo_key is None:
                raise ValueError("CSV must contain headers for subject and photo (e.g., subject_id, photo_id).")
            mapping[str(row[photo_key])] = str(row[subj_key])
    return mapping


# -------------------- Main training function --------------------
def main():
    parser = argparse.ArgumentParser("MNASNet Body Measurement Training (Frozen Backbone)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subject_map_csv", type=str, default=None)

    parser.add_argument("--out_dir", type=str, default="runs/mnasnet_bm")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/mnasnet_bm")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1",
                        choices=["IMAGENET1K_V1", "DEFAULT", "NONE"])

    parser.add_argument("--single_h", type=int, default=640)
    parser.add_argument("--single_w", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_iters", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_model_path", type=str, default=None)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    split_dir = Path(args.data_root) / args.split
    samples, h_min, h_max, w_min, w_max = build_samples(split_dir)
    dataset = BodyMDataset(samples, args.single_h, args.single_w)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True)

    # Model
    model = MNASNetRegressor(num_outputs=14, weights=args.weights).to(device)

    # Freeze backbone completely
    backbone_params, head_params = group_params(model)
    for p in backbone_params:
        p.requires_grad = False
    for p in head_params:
        p.requires_grad = True

    # Optimizer (only head parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.L1Loss(reduction="mean")  # MAE loss

    writer = SummaryWriter(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    iters_per_epoch = ceil(len(dataset) / args.batch_size)
    max_iters = args.max_iters
    reduce_iters = [int(0.75 * max_iters), int(0.88 * max_iters)]

    run_cfg = RunConfig(
        data_root=args.data_root,
        split=args.split,
        single_h=args.single_h,
        single_w=args.single_w,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
         max_iters=max_iters,
        reduce_iters=reduce_iters,
        out_dir=args.out_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    save_run_config(run_cfg, args.out_dir)

    subject_map = read_subject_map_csv(Path(args.subject_map_csv)) if args.subject_map_csv else None

    # ---------------- Evaluation-only mode ----------------
    if args.eval_only:
        if args.eval_model_path:
            sd = torch.load(args.eval_model_path, map_location="cpu")
            model.load_state_dict(sd if isinstance(sd, dict) else sd["model_state_dict"])
        per_m_mae, overall_mae, tp = evaluate_subjectwise_model(
            model, samples, device, subject_map=subject_map
        )
        print("\n=== Evaluation (subject-wise) ===")
        for i, m in enumerate(per_m_mae):
            print(f"MAE[{i:02d}]: {m:.3f} mm")
        print(f"Overall MAE: {overall_mae:.3f} mm")
        print(f"TP50: {tp['TP50']:.3f}  TP75: {tp['TP75']:.3f}  TP90: {tp['TP90']:.3f}\n")
        return

    # ---------------- Training ----------------
    iteration = 0
    epoch = 0
    best_overall = float("inf")

    while iteration < max_iters:
        model.train()
        running_loss = 0.0
        batch_count = 0

        for x, y, sid, _meta in loader:
            if iteration >= max_iters:
                break

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                preds = model(x)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
            batch_count += 1
            iteration += 1

            # LR reductions at milestones
            if iteration in reduce_iters:
                new_lrs = reduce_lr_by_factor(optimizer, factor=0.1)
                writer.add_scalar("train/lr", new_lrs[0], global_step=iteration)

        epoch += 1
        avg_train_loss = running_loss / max(1, batch_count)
        writer.add_scalar("train/loss_mae_mm", avg_train_loss, global_step=iteration)

        # Validation
        if epoch % args.eval_every == 0:
            eval_subset = random.sample(samples, k=min(256, len(samples))) if len(samples) > 1000 else samples
            per_m_mae, overall_mae, tp = evaluate_subjectwise_model(
                model, eval_subset, device, subject_map=subject_map
            )

            writer.add_scalar("val/overall_mae_mm", overall_mae, global_step=iteration)
            writer.add_text("val/TP", str(tp), global_step=iteration)
            for i, m in enumerate(per_m_mae):
                writer.add_scalar(f"val/mae_mm_{i:02d}", m, global_step=iteration)

            tp_str = f"TP50={tp['TP50']:.3f}  TP75={tp['TP75']:.3f}  TP90={tp['TP90']:.3f}"
            print(f"[Epoch {epoch:03d} | Iter {iteration:07d}] "
                  f"train_mae={avg_train_loss:.4f}  val_overall_mae={overall_mae:.4f}  {tp_str}")

            ckpt = {
                "epoch": epoch,
                "iteration": iteration,
                "loss": avg_train_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))

            if overall_mae < best_overall:
                best_overall = overall_mae
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_mnasnet_bmnet.pt"))

        if iteration >= max_iters:
            break

    print(f"\nBest validation overall MAE: {best_overall:.3f} mm")


if __name__ == "__main__":
    main()


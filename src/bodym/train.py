import os
import random
import torch
import numpy as np
from math import ceil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import collections

from bodym.data import BodyMDataset, build_samples
from bodym.models import MNASNetRegressor
from bodym.utils import RunConfig, seed_everything, save_run_config, reduce_lr_by_factor


def evaluate_subjectwise_model(model, sample_list, device):
    model.eval()
    subj_to_errs = {}

    with torch.no_grad():
        for s in sample_list:
            single_ds = BodyMDataset([s])
            loader = DataLoader(single_ds, batch_size=1, shuffle=False, num_workers=0)
            for x, y, sid in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                err = torch.abs(pred - y).cpu().numpy().reshape(-1)
                sid = sid[0]
                subj_to_errs.setdefault(sid, []).append(err)

    subj_maes = np.array([np.mean(subj_to_errs[k], axis=0) for k in subj_to_errs.keys()])
    per_measurement_mae = np.mean(subj_maes, axis=0)
    overall_mae = np.mean(per_measurement_mae)
    tp_dict = {
        "TP50": np.mean(np.percentile(subj_maes, 50, axis=0)),
        "TP75": np.mean(np.percentile(subj_maes, 75, axis=0)),
        "TP90": np.mean(np.percentile(subj_maes, 90, axis=0)),
    }
    return per_measurement_mae, overall_mae, tp_dict


def main(args):
    # ---------------- Setup ----------------
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    split_dir = os.path.join(args.data_root, args.split)
    samples, h_min, h_max, w_min, w_max = build_samples(split_dir)

    ### ---------------- SUBJECT-WISE 90/10 SPLIT (NEW) ---------------- ###
    # Only applied when using training data
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
        val_samples = samples  # for eval-only or test splits
    ### ------------------------------------------------------------------ ###

    dataset = BodyMDataset(train_samples, args.single_h, args.single_w)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------------- Model ----------------
    model = MNASNetRegressor(num_outputs=14, weights=args.weights).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    loss_fn = torch.nn.L1Loss()
    writer = SummaryWriter(args.out_dir)
    scaler = GradScaler(enabled=(device == "cuda"))

    # ---------------- LR and Iterations ----------------
    iters_per_epoch = ceil(len(dataset) / args.batch_size)
    max_iters = args.max_iters
    reduce_iters = [int(0.75 * max_iters), int(0.88 * max_iters)]
    num_epochs = ceil(max_iters / iters_per_epoch)

    # ---------------- Save Run Config ----------------
    run_config = RunConfig(
        data_root=args.data_root,
        split=args.split,
        single_h=args.single_h,
        single_w=args.single_w,
        batch_size=args.batch_size,
        max_iters=max_iters,
        lr=args.lr,
        weights=args.weights,
        seed=args.seed,
        out_dir=args.out_dir,
        checkpoint_dir=args.checkpoint_dir,
        reduce_iters=reduce_iters,
        num_epochs=num_epochs,
    )
    save_run_config(run_config, args.out_dir)

    # ---------------- Eval Only ----------------
    if args.eval_only:
        per_measurement_mae, overall_mae, tp_dict = evaluate_subjectwise_model(
            model, val_samples, device
        )
        print(f"Eval-only mode on {args.split} set:")
        print(f"Overall MAE (mm): {overall_mae:.2f}")
        print(f"TP metrics: {tp_dict}")
        return

    # ---------------- Training Loop ----------------
    iteration = 0
    best_val_mae = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for x_batch, y_batch, sid_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device == "cuda")):
                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1
            iteration += 1

            # Learning rate schedule
            if iteration in reduce_iters:
                reduce_lr_by_factor(optimizer, 0.1)
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/lr", current_lr, iteration)
                print(f"LR reduced to {current_lr:.6f} at iteration {iteration}")

            if iteration >= max_iters:
                break

        avg_train_loss = running_loss / max(1, batch_count)

        # ---------------- Validation ----------------
        model.eval()
        if args.split == "train":
            val_subset = val_samples if len(val_samples) <= 1000 else random.sample(val_samples, 256)
        else:
            val_subset = val_samples

        per_measurement_mae, overall_mae, tp_dict = evaluate_subjectwise_model(
            model, val_subset, device
        )

        writer.add_scalar("train/loss", avg_train_loss, iteration)
        writer.add_scalar("val/overall_mae_mm", overall_mae, iteration)
        writer.add_text("val/TP", str(tp_dict), iteration)

        print(
            f"Epoch [{epoch+1}/{num_epochs}]  Iter {iteration}/{max_iters} "
            f"TrainLoss={avg_train_loss:.4f}  ValMAE={overall_mae:.2f}mm  TP={tp_dict}"
        )

        # ---------------- Checkpoints ----------------
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "iteration": iteration,
                "loss": avg_train_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )

        if overall_mae < best_val_mae:
            best_val_mae = overall_mae
            best_path = os.path.join(args.checkpoint_dir, "best_mnasnet_bmnet.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved at {best_path} (MAE={best_val_mae:.2f}mm)")

        if iteration >= max_iters:
            break

    print(f"Training complete. Best validation MAE: {best_val_mae:.2f} mm")


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MNASNet on BodyM dataset")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--single_h", type=int, default=640)
    parser.add_argument("--single_w", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=22)
    parser.add_argument("--max_iters", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1", choices=["IMAGENET1K_V1", "DEFAULT", "NONE"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--out_dir", type=str, default="runs/mnasnet_bodym")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/mnasnet_bodym")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)


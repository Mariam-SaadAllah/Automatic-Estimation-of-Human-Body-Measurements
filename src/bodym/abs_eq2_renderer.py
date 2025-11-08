#!/usr/bin/env python3
"""
Render SMPL-X silhouettes for frontal (0°) and lateral (90°) views,
matching Eq. (2) input (two-view binary masks) for the BodyM pipeline.

Outputs:
  outputs/mask/<subject_id>.png
  outputs/mask_left/<subject_id>.png
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# --- Only SMPL-X ---
import smplx  # pip install smplx


def rotation_matrix_y(deg: float) -> np.ndarray:
    """Rotation matrix around the Y-axis."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float32)


def orthographic_project(verts: np.ndarray, img_h: int, img_w: int,
                         pad: float = 0.05, scale: float | None = None) -> tuple[np.ndarray, float]:
    """
    Orthographic projection to image plane. If scale is None, compute scale
    to tightly fit the mesh; otherwise, reuse the provided scale (for consistent views).
    Returns (2D points, scale).
    """
    v = verts - verts.mean(axis=0, keepdims=True)

    xyz_max = np.abs(v).max()
    if scale is None:
        if xyz_max < 1e-8:
            scale = 1.0
        else:
            scale = 1.0 / (xyz_max * (1.0 + pad))

    v = v * scale

    # Map x->cols, y->rows (flip y for image coordinates)
    x = v[:, 0]
    y = -v[:, 1]
    # Normalize to [0, 1]
    x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y01 = (y - y.min()) / (y.max() - y.min() + 1e-8)
    # Convert to pixel coordinates
    px = (x01 * (img_w - 1)).astype(np.int32)
    py = (y01 * (img_h - 1)).astype(np.int32)
    pts = np.stack([px, py], axis=1)
    return pts, scale


def rasterize_silhouette(pts2d: np.ndarray, faces: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Rasterize mesh faces into a binary silhouette image."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    polys = [pts2d[f].reshape(-1, 1, 2) for f in faces]
    cv2.fillPoly(mask, polys, color=255, lineType=cv2.LINE_8, shift=0)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Render SMPL-X silhouettes (front + side) from betas ~ 0.")
    parser.add_argument("--model_path", type=str,
                        default=r"C:\Users\Mariam\Desktop\SMPL-X\models\smplx\SMPLX_NEUTRAL.npz",
                        help="Path to SMPLX_NEUTRAL.npz")
    parser.add_argument("--subject_id", type=str, default="subj_0001", help="Output subject id (filename stem)")
    parser.add_argument("--gender", type=str, default="NEUTRAL", choices=["NEUTRAL", "MALE", "FEMALE"])
    parser.add_argument("--num_betas", type=int, default=10)
    parser.add_argument("--beta_std", type=float, default=0.03, help="Std dev for betas ~ N(0, beta_std^2)")
    parser.add_argument("--img_h", type=int, default=640)
    parser.add_argument("--img_w", type=int, default=480)
    parser.add_argument("--out_root", type=str, default="outputs", help="Root folder to write mask & mask_left")
    parser.add_argument("--show", action="store_true", help="Show the two silhouettes in a window")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_root)
    out_mask = out_root / "mask"
    out_mask_left = out_root / "mask_left"
    out_mask.mkdir(parents=True, exist_ok=True)
    out_mask_left.mkdir(parents=True, exist_ok=True)

    # --- Build SMPL-X model (neutral pose) ---
    model = smplx.SMPLX(
        model_path=str(Path(args.model_path).parent),
        model_type="smplx",
        gender=args.gender.lower(),
        num_betas=args.num_betas,
        use_pca=False,
        flat_hand_mean=True
    ).to(device)

    # --- Generate random shape parameters ---
    with torch.no_grad():
        betas = torch.randn(1, args.num_betas, device=device) * args.beta_std
        global_orient = torch.zeros(1, 3, device=device)
        body_pose = torch.zeros(1, 21 * 3, device=device)
        left_hand_pose = torch.zeros(1, 15 * 3, device=device)
        right_hand_pose = torch.zeros(1, 15 * 3, device=device)
        jaw_pose = torch.zeros(1, 3, device=device)
        leye_pose = torch.zeros(1, 3, device=device)
        reye_pose = torch.zeros(1, 3, device=device)
        expression = torch.zeros(1, 10, device=device)

        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression,
            return_verts=True
        )

        verts = output.vertices[0].detach().cpu().numpy()
        faces = model.faces.astype(np.int32)

    # --- Render two views with shared scale ---
    H, W = args.img_h, args.img_w

    # Frontal
    R_front = rotation_matrix_y(0.0)
    verts_front = (R_front @ verts.T).T
    pts_front, scale = orthographic_project(verts_front, H, W)

    front_mask = rasterize_silhouette(pts_front, faces, H, W)

    # Lateral (reuse scale)
    R_side = rotation_matrix_y(90.0)
    verts_side = (R_side @ verts.T).T
    pts_side, _ = orthographic_project(verts_side, H, W, scale=scale)

    side_mask = rasterize_silhouette(pts_side, faces, H, W)

    # --- Save to dataset-compatible folders ---
    front_path = out_mask / f"{args.subject_id}.png"
    side_path = out_mask_left / f"{args.subject_id}.png"
    cv2.imwrite(str(front_path), front_mask)
    cv2.imwrite(str(side_path), side_mask)

    print(f"Saved:\n  {front_path}\n  {side_path}")

    # --- Optional visualization ---
    if args.show:
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        axes[0].imshow(front_mask, cmap="gray")
        axes[0].set_title("Frontal (0°)")
        axes[0].axis("off")
        axes[1].imshow(side_mask, cmap="gray")
        axes[1].set_title("Lateral (90°)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()



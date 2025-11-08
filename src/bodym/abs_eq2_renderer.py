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
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=np.float32)


def orthographic_project(verts: np.ndarray, img_h: int, img_w: int, pad: float = 0.05) -> np.ndarray:
    """
    Simple orthographic projection that fits the whole mesh tightly into the image,
    preserving aspect ratio. Returns 2D points in pixel coords (x,y).
    """
    # Center to origin
    v = verts - verts.mean(axis=0, keepdims=True)

    # Scale to fit image with small padding
    xyz_max = np.abs(v).max()
    if xyz_max < 1e-8:
        s = 1.0
    else:
        s = 1.0 / (xyz_max * (1.0 + pad))

    v = v * s
    # Map x->cols, y->rows (flip y-axis to image coordinates)
    x = v[:, 0]
    y = -v[:, 1]  # image y goes downward
    # Normalize to [0,1]
    x01 = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y01 = (y - y.min()) / (y.max() - y.min() + 1e-8)
    # To pixels
    px = (x01 * (img_w - 1)).astype(np.int32)
    py = (y01 * (img_h - 1)).astype(np.int32)
    pts = np.stack([px, py], axis=1)
    return pts


def rasterize_silhouette(pts2d: np.ndarray, faces: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    Fills all projected triangles onto a blank image to produce a binary silhouette.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # Prepare list of contours for cv2.fillPoly
    polys = []
    for f in faces:
        tri = pts2d[f]  # (3,2)
        polys.append(tri.reshape(-1, 1, 2))
    cv2.fillPoly(mask, polys, color=255, lineType=cv2.LINE_8, shift=0)
    return mask


def render_view(verts: np.ndarray, faces: np.ndarray, yaw_deg: float, H: int, W: int) -> np.ndarray:
    R = rotation_matrix_y(yaw_deg)
    v_rot = (R @ verts.T).T  # (V,3)
    pts = orthographic_project(v_rot, H, W)
    mask = rasterize_silhouette(pts, faces, H, W)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Render SMPL-X silhouettes (front + side) from betas ~ 0.")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\Mariam\Desktop\SMPL-X\models\smplx\SMPLX_NEUTRAL.npz",
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

    # --- Build SMPL-X model (only body, neutral pose) ---
    model = smplx.SMPLX(
        model_path=str(Path(args.model_path).parent),  # directory containing SMPLX_*.npz
        model_type="smplx",
        gender=args.gender.lower(),  # 'neutral', 'male', 'female'
        num_betas=args.num_betas,
        use_pca=False,             # no PCA for hands
        flat_hand_mean=True
    ).to(device)

    with torch.no_grad():
        betas = torch.randn(1, args.num_betas, device=device) * args.beta_std
        global_orient = torch.zeros(1, 3, device=device)
        body_pose = torch.zeros(1, 21 * 3, device=device)       # 21 joints * 3 axis-angle
        left_hand_pose = torch.zeros(1, 15 * 3, device=device)  # 15 joints * 3
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

        verts = output.vertices[0].detach().cpu().numpy()  # (V,3)
        faces = model.faces.astype(np.int32)               # (F,3)

    # --- Render two views: frontal (0°) and lateral (90°) ---
    H, W = args.img_h, args.img_w
    front = render_view(verts, faces, yaw_deg=0.0, H=H, W=W)
    side = render_view(verts, faces, yaw_deg=90.0, H=H, W=W)

    # --- Save to dataset-compatible folders ---
    front_path = out_mask / f"{args.subject_id}.png"
    side_path = out_mask_left / f"{args.subject_id}.png"
    cv2.imwrite(str(front_path), front)
    cv2.imwrite(str(side_path), side)

    print(f"Saved:\n  {front_path}\n  {side_path}")

    if args.show:
        # Quick visual check
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        axes[0].imshow(front, cmap="gray")
        axes[0].set_title("Frontal (0°)")
        axes[0].axis("off")
        axes[1].imshow(side, cmap="gray")
        axes[1].set_title("Lateral (90°)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Render SMPL-X silhouettes for frontal (0°) and lateral (90°) views
using PyTorch3D. Matches Eq. (2) in the Human Body Measurement Estimation paper.

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

import smplx
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftSilhouetteShader,
    look_at_view_transform,
)

def main():
    parser = argparse.ArgumentParser(description="Render SMPL-X silhouettes (front + side) using PyTorch3D.")
    parser.add_argument("--model_path", type=str,
                        default="/content/drive/MyDrive/BMNet_Project/smplx/SMPLX_NEUTRAL.npz",
                        help="Path to SMPLX_NEUTRAL.npz or other gender model file.")
    parser.add_argument("--subject_id", type=str, default="subj_0001",
                        help="Output subject ID (filename stem).")
    parser.add_argument("--gender", type=str, default="NEUTRAL",
                        choices=["NEUTRAL", "MALE", "FEMALE"])
    parser.add_argument("--num_betas", type=int, default=10)
    parser.add_argument("--beta_std", type=float, default=0.03,
                        help="Std dev for betas ~ N(0, beta_std^2)")
    parser.add_argument("--img_h", type=int, default=640)
    parser.add_argument("--img_w", type=int, default=480)
    parser.add_argument("--out_root", type=str, default="outputs",
                        help="Output directory root")
    parser.add_argument("--show", action="store_true",
                        help="Show rendered silhouettes.")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Setup device and output folders
    # ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_root)
    out_mask = out_root / "mask"
    out_mask_left = out_root / "mask_left"
    out_mask.mkdir(parents=True, exist_ok=True)
    out_mask_left.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load SMPL-X model (neutral pose)
    # ---------------------------------------------------------------
    model = smplx.SMPLX(
        model_path=str(Path(args.model_path).parent),
        model_type="smplx",
        gender=args.gender.lower(),
        num_betas=args.num_betas,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    # ---------------------------------------------------------------
    # Generate random shape parameters (betas)
    # ---------------------------------------------------------------
    with torch.no_grad():
        betas = torch.randn(1, args.num_betas, device=device) * args.beta_std
        output = model(
            betas=betas,
            body_pose=torch.zeros(1, 21 * 3, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            left_hand_pose=torch.zeros(1, 15 * 3, device=device),
            right_hand_pose=torch.zeros(1, 15 * 3, device=device),
            expression=torch.zeros(1, 10, device=device),
            return_verts=True,
        )

    # ---------------------------------------------------------------
    # Prepare mesh for PyTorch3D
    # ---------------------------------------------------------------
    verts = output.vertices[0].unsqueeze(0)  # shape (1, N_v, 3)
    faces = torch.tensor(model.faces.astype(np.int64), device=device).unsqueeze(0)
    mesh = Meshes(verts=verts, faces=faces)

    # ---------------------------------------------------------------
    # Rendering setup
    # ---------------------------------------------------------------
    raster_settings = RasterizationSettings(
        image_size=(args.img_h, args.img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    def render_silhouette(yaw_deg: float, out_path: Path):
        """Render a silhouette at a specific yaw angle and save it."""
        # Camera setup
        R, T = look_at_view_transform(dist=2.0, elev=0.0, azim=yaw_deg)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader()
        )

        # Forward pass
        silhouette = renderer(mesh, cameras=cameras)
        mask = (silhouette[0, ..., 3] > 0.5).cpu().numpy().astype(np.uint8) * 255
        cv2.imwrite(str(out_path), mask)
        return mask

    # ---------------------------------------------------------------
    # Render both frontal (0°) and lateral (90°) views
    # ---------------------------------------------------------------
    front_path = out_mask / f"{args.subject_id}.png"
    side_path = out_mask_left / f"{args.subject_id}.png"
    mask_front = render_silhouette(0.0, front_path)
    mask_side = render_silhouette(90.0, side_path)

    print(f" Saved silhouettes:\n  {front_path}\n  {side_path}")

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------
    if args.show:
        fig, ax = plt.subplots(1, 2, figsize=(8, 6))
        ax[0].imshow(mask_front, cmap="gray")
        ax[0].set_title("Frontal (0°)")
        ax[0].axis("off")

        ax[1].imshow(mask_side, cmap="gray")
        ax[1].set_title("Lateral (90°)")
        ax[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()





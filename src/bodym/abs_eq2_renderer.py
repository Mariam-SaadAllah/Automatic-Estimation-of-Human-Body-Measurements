#!/usr/bin/env python3
"""
Generate SMPL-X silhouettes for frontal (0°) and lateral (90°) views,
reproducing Eq. (2) input from:
  "Human Body Measurement Estimation with Adversarial Augmentation"

Outputs:
  outputs/mask/<subject_id>.png
  outputs/mask_left/<subject_id>.png

β is sampled from a small Gaussian distribution around zero
(β ~ N(0, β_std²)) to produce random but realistic body shapes.
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


# ---------------------------------------------------------------------
# Helper function: render one binary silhouette at a given yaw angle
# ---------------------------------------------------------------------
def render_silhouette(mesh: Meshes, yaw_deg: float, H: int, W: int, device: str, out_path: Path) -> np.ndarray:
    """
    Render a binary silhouette from a given yaw rotation angle
    using PyTorch3D's soft silhouette renderer.

    Parameters
    ----------
    mesh : Meshes
        SMPL-X mesh to render.
    yaw_deg : float
        Yaw rotation angle in degrees (0° = frontal, 90° = lateral).
    H, W : int
        Image height and width.
    device : str
        'cuda' or 'cpu'.
    out_path : Path
        Where to save the output binary mask.

    Returns
    -------
    np.ndarray
        Binary mask of shape (H, W), white body on black background.
    """
    # Camera placed at distance 2.0 units away, rotated around Y axis by yaw_deg
    R, T = look_at_view_transform(dist=2.0, elev=0.0, azim=yaw_deg)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Rasterizer and renderer configuration
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,  # disable binning (avoids overflow warnings for large meshes)
        max_faces_per_bin=150000,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader()
    )

    # Forward pass: RGBA silhouette image
    rgba = renderer(mesh, cameras=cameras)  # (1, H, W, 4)
    mask = (rgba[0, ..., 3] > 0.5).detach().cpu().numpy().astype(np.uint8) * 255
    cv2.imwrite(str(out_path), mask)
    return mask


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Render SMPL-X silhouettes (front + side) using PyTorch3D.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/content/drive/MyDrive/BMNet_Project/smplx/SMPLX_NEUTRAL.npz",
        help="Path to a SMPL-X model file (.npz or .pkl)."
    )
    parser.add_argument("--subject_id", type=str, default="subj_0001", help="Output subject identifier (filename stem).")
    parser.add_argument("--gender", type=str, default="NEUTRAL", choices=["NEUTRAL", "MALE", "FEMALE"], help="SMPL-X gender model.")
    parser.add_argument("--num_betas", type=int, default=10, help="Number of shape coefficients (β).")
    parser.add_argument("--beta_std", type=float, default=0.03, help="Standard deviation for β ~ N(0, β_std²).")
    parser.add_argument("--img_h", type=int, default=640, help="Silhouette image height (default: 640).")
    parser.add_argument("--img_w", type=int, default=480, help="Silhouette image width (default: 480).")
    parser.add_argument("--out_root", type=str, default="outputs", help="Root output directory.")
    parser.add_argument("--show", action="store_true", help="Display the generated silhouettes.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate model path
    model_file = Path(args.model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"SMPL-X model file not found:\n  {model_file}")
    smpl_dir = str(model_file.parent)

    # Output directory structure
    out_root = Path(args.out_root)
    out_mask = out_root / "mask"
    out_mask_left = out_root / "mask_left"
    out_mask.mkdir(parents=True, exist_ok=True)
    out_mask_left.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Load SMPL-X model
    # -----------------------------------------------------------------
    model = smplx.SMPLX(
        model_path=smpl_dir,
        model_type="smplx",
        gender=args.gender.lower(),
        num_betas=args.num_betas,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    # -----------------------------------------------------------------
    # 2) Sample shape coefficients β ~ N(0, β_std²)
    # -----------------------------------------------------------------
    with torch.no_grad():
        betas = torch.randn(1, args.num_betas, device=device) * args.beta_std

        # Neutral pose and expression
        output = model(
            betas=betas,
            body_pose=torch.zeros(1, 21 * 3, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            left_hand_pose=torch.zeros(1, 15 * 3, device=device),
            right_hand_pose=torch.zeros(1, 15 * 3, device=device),
            expression=torch.zeros(1, 10, device=device),
            return_verts=True,
        )

    # -----------------------------------------------------------------
    # 3) Construct a PyTorch3D mesh
    # -----------------------------------------------------------------
    verts = output.vertices[0].unsqueeze(0)  # (1, V, 3)
    faces = torch.as_tensor(model.faces.astype(np.int64), device=device).unsqueeze(0)  # (1, F, 3)
    mesh = Meshes(verts=verts, faces=faces)

    # -----------------------------------------------------------------
    # 4) Render frontal and lateral silhouettes
    # -----------------------------------------------------------------
    H, W = args.img_h, args.img_w
    front_path = out_mask / f"{args.subject_id}.png"
    side_path = out_mask_left / f"{args.subject_id}.png"

    mask_front = render_silhouette(mesh, yaw_deg=0.0, H=H, W=W, device=device, out_path=front_path)
    mask_side = render_silhouette(mesh, yaw_deg=90.0, H=H, W=W, device=device, out_path=side_path)

    print(f"Saved silhouettes:\n  {front_path}\n  {side_path}")

    # -----------------------------------------------------------------
    # 5) Optional: visualization for inspection
    # -----------------------------------------------------------------
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






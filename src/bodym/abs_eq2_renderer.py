#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eq.(2) renderer for SMPL-X:
  x = R(M(β, θ), ι, γ)
Generates N samples with:
  - SMPL-X mesh in A-pose (θ = 0)
  - β ~ N(0, BETA_STD^2)
  - Two silhouettes (front & side) with a perspective camera at ~1.8 m
  - Exports .obj meshes and a CSV manifest

Usage:
  python abs_eq2_renderer.py \
    --model_path "/content/drive/MyDrive/BMNet_Project/smplx/SMPLX_NEUTRAL.npz" \
    --out_dir eq2_outputs --num 3
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import smplx
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRenderer, MeshRasterizer,
    RasterizationSettings, SoftSilhouetteShader, look_at_view_transform
)
import cv2


def resolve_model_dir(model_path: str | Path) -> Path:
    p = Path(model_path)
    if p.is_file():
        return p.parent
    return p


def build_smplx_mesh(model_dir: Path, device: str, beta_std: float) -> tuple[Meshes, torch.Tensor, torch.Tensor]:
    model = smplx.create(
        model_path=str(model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True
    ).to(device)

    betas = beta_std * torch.randn(1, 10, device=device)
    body_pose = torch.zeros(1, 21*3, device=device)  # A-pose (zeros in SMPL-X)
    global_orient = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        out = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            return_verts=True
        )

    verts = out.vertices[0]
    faces = torch.from_numpy(model.faces.astype(np.int64)).to(device)
    mesh = Meshes(verts=[verts], faces=[faces])
    return mesh, verts, faces


def make_renderer(device: str, img_size: int, fov_deg: float, R, T):
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov_deg)
    raster = RasterizationSettings(image_size=img_size)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftSilhouetteShader()
    )


def render_silhouette(mesh: Meshes, device: str, camera_dist: float, fov_deg: float, img_size: int, azim_deg: float) -> np.ndarray:
    R, T = look_at_view_transform(dist=camera_dist, elev=0.0, azim=azim_deg, device=device)
    renderer = make_renderer(device, img_size, fov_deg, R, T)
    rgba = renderer(mesh)
    alpha = rgba[0, ..., 3].clamp(0, 1).detach().cpu().numpy()
    img = (alpha * 255).astype(np.uint8)
    return img


def save_png_gray(path: Path, img: np.ndarray):
    cv2.imwrite(str(path), img)


def main():
    parser = argparse.ArgumentParser(description="SMPL-X Eq.(2) silhouette renderer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to SMPL-X model .npz file OR the directory containing it")
    parser.add_argument("--out_dir", type=str, default="eq2_outputs", help="Output directory")
    parser.add_argument("--num", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--img_size", type=int, default=512, help="Silhouette image size (square)")
    parser.add_argument("--camera_dist", type=float, default=1.8, help="Camera distance in meters (paper ~1.68–1.98 m)")
    parser.add_argument("--fov_deg", type=float, default=60.0, help="Camera field of view in degrees")
    parser.add_argument("--beta_std", type=float, default=0.03, help="Std-dev for shape betas (around zero)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model_dir(args.model_path)

    records = []
    for i in range(args.num):
        mesh, verts, faces = build_smplx_mesh(model_dir, device, args.beta_std)

        obj_path = out_dir / f"sample_{i+1:03d}.obj"
        save_obj(obj_path, verts.cpu(), faces.cpu())

        sil_front = render_silhouette(mesh, device, args.camera_dist, args.fov_deg, args.img_size, azim_deg=0.0)
        sil_side  = render_silhouette(mesh, device, args.camera_dist, args.fov_deg, args.img_size, azim_deg=90.0)

        front_path = out_dir / f"sample_{i+1:03d}_sil_front.png"
        side_path  = out_dir / f"sample_{i+1:03d}_sil_side.png"
        save_png_gray(front_path, sil_front)
        save_png_gray(side_path,  sil_side)

        records.append({
            "sample_id": i+1,
            "obj_path": str(obj_path),
            "sil_front_path": str(front_path),
            "sil_side_path": str(side_path),
            "camera_distance_m": args.camera_dist,
            "fov_deg": args.fov_deg,
            "image_size_px": args.img_size,
            "azim_front_deg": 0.0,
            "azim_side_deg": 90.0,
            "beta_std": args.beta_std,
            "pose": "A-pose (zeros)",
            "renderer": "PyTorch3D SoftSilhouetteShader, FoV perspective"
        })

    # Write CSV manifest
    csv_path = out_dir / "eq2_manifest.csv"
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)

    print(f"Done. Wrote {len(records)} samples to {out_dir}")
    print(f"CSV manifest: {csv_path}")


if __name__ == "__main__":
    main()

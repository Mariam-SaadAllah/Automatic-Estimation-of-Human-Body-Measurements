"""
abs_eq2_renderer.py
------------------------------------------------------------
Implements Section 3.2, Eq. (2) of:
  "Human Body Measurement Estimation with Adversarial Augmentation"

Equation:
    x = R(M(β, θ), ι, γ)

Where:
  M(β, θ)  -> SMPL-X body mesh generated from shape (β) and pose (θ)
  ι (iota) -> lighting parameters
  γ (gamma)-> camera parameters (frontal & lateral)
  R(·)     -> graphics renderer (PyTorch3D)
  x        -> 2-D silhouette image

Outputs (default 640×480, as in the paper):
  - render_front_silhouette.png
  - render_side_silhouette.png
  - render_front_mesh.png
  - render_side_mesh.png
  - silhouette_combined_480x960.png  (front|side concatenated for BMnet)

Usage (local):
  python abs_eq2_renderer.py \
      --model_dir assets/smplx_models \
      --out_dir outputs \
      --height 480 --width 640
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch
import smplx
import numpy as np
import imageio.v2 as imageio

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    HardPhongShader,
    OrthographicCameras,
    PointLights,
    TexturesVertex,
    look_at_view_transform,
)


def build_smplx_mesh(model_dir: Path, device: torch.device):
    """
    Create a neutral SMPL-X mesh in A-pose:
      - β ≈ 0 (average shape)
      - θ = 0 (A-pose), global_orient = 0
    Returns:
      verts (V,3), faces (F,3)
    """
    model = smplx.create(
        model_path=str(model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
    ).to(device)

    betas = torch.zeros(1, 10, device=device)  # shape around 0
    body_pose = torch.zeros(1, model.NUM_BODY_JOINTS * 3, device=device)  # A-pose
    global_orient = torch.zeros(1, 3, device=device)

    out = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        return_verts=True,
    )
    verts = out.vertices[0]  # (V,3)
    faces = torch.tensor(model.faces.astype(int), device=device)  # (F,3)
    return verts, faces


def make_cameras(device: torch.device, dist=2.5, elev=0.0):
    """
    Two Orthographic cameras:
      - frontal: azim=0°
      - lateral: azim=90°
    """
    Rf, Tf = look_at_view_transform(dist=dist, elev=elev, azim=0.0, device=device)
    Rs, Ts = look_at_view_transform(dist=dist, elev=elev, azim=90.0, device=device)
    cam_front = OrthographicCameras(device=device, R=Rf, T=Tf)
    cam_side  = OrthographicCameras(device=device, R=Rs, T=Ts)
    return cam_front, cam_side


def make_renderers(image_h: int, image_w: int, device: torch.device, cam_for_shaded):
    """
    Create two renderers:
      - silhouette (SoftSilhouetteShader) -> masks for BMnet
      - shaded (HardPhongShader)          -> mesh visualization
    """
    raster_settings = RasterizationSettings(
        image_size=(image_h, image_w),  # (H, W) = (480, 640)
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer_sil = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(),
    )

    renderer_shaded = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cam_for_shaded),
    )

    return renderer_sil, renderer_shaded


def save_png(path: Path, arr_float01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), (np.clip(arr_float01, 0, 1) * 255).astype("uint8"))


def main(args):
    # ---------------- Env ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    H, W = args.height, args.width  # 480×640 per paper

    # -------------- Mesh -----------------
    verts, faces = build_smplx_mesh(model_dir, device)

    # simple gray vertex color (only for shaded render)
    verts_rgb = torch.ones_like(verts)[None] * 0.7
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=verts_rgb),
    )

    # -------- Lighting & Cameras ---------
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    cam_front, cam_side = make_cameras(device, dist=args.cam_dist, elev=args.cam_elev)

    # -------- Renderers (640×480) --------
    renderer_sil, renderer_shaded = make_renderers(H, W, device, cam_for_shaded=cam_front)

    # ---- Eq.(2): x = R(M(β,θ), ι, γ) ----
    sil_front = renderer_sil(meshes_world=mesh, cameras=cam_front, lights=lights)[0, ..., 3].detach().cpu().numpy()
    sil_side  = renderer_sil(meshes_world=mesh, cameras=cam_side,  lights=lights)[0, ..., 3].detach().cpu().numpy()

    mesh_front = renderer_shaded(meshes_world=mesh, cameras=cam_front, lights=lights)[0, ..., :3].detach().cpu().numpy()
    mesh_side  = renderer_shaded(meshes_world=mesh, cameras=cam_side,  lights=lights)[0, ..., :3].detach().cpu().numpy()

    # -------------- Save -----------------
    save_png(out_dir / "render_front_silhouette.png", sil_front)
    save_png(out_dir / "render_side_silhouette.png",  sil_side)
    save_png(out_dir / "render_front_mesh.png",       mesh_front)
    save_png(out_dir / "render_side_mesh.png",        mesh_side)

    # concatenate silhouettes horizontally -> (480 × 960), as Section 3.1 input
    combined = np.concatenate([sil_front, sil_side], axis=1)
    save_png(out_dir / "silhouette_combined_480x960.png", combined)

    print(" Saved to:", out_dir.resolve())
    print("  - render_front_silhouette.png")
    print("  - render_side_silhouette.png")
    print("  - render_front_mesh.png")
    print("  - render_side_mesh.png")
    print("  - silhouette_combined_480x960.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABS Eq.(2) renderer (SMPL-X + PyTorch3D)")
    parser.add_argument("--model_dir", type=str, default="assets/smplx_models",
                        help="Directory with SMPLX_NEUTRAL.npz")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="Directory to save renders")
    parser.add_argument("--height", type=int, default=480, help="Output height (paper: 480)")
    parser.add_argument("--width",  type=int, default=640, help="Output width  (paper: 640)")
    parser.add_argument("--cam_dist", type=float, default=2.5, help="Camera distance")
    parser.add_argument("--cam_elev", type=float, default=0.0, help="Camera elevation (deg)")
    args = parser.parse_args()
    main(args)

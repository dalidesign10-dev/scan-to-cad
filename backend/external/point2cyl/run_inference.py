"""Standalone Point2Cyl inference runner.

Usage:
    python run_inference.py <mesh_path> <output_json>

Loads the pretrained Point2Cyl model from results/Point2Cyl_without_sketch/model.pth,
samples 8192 points from the mesh, runs inference, and writes a JSON file with
segmented extrusion cylinders ready for the GeoMagic Claude frontend.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from models.pointnet_extrusion import backbone

K = 8  # Max extrusions (matches training)
NUM_POINT = 8192
# Pick checkpoint via env var so the server can A/B test variants:
#   POINT2CYL_CKPT = "without_sketch" (default) | "deepcad" | "with_sketch"
_CKPT_NAME = os.environ.get("POINT2CYL_CKPT", "without_sketch").lower()
_CKPT_PATHS = {
    "without_sketch": ("Point2Cyl_without_sketch", "model.pth"),
    "deepcad":        ("Point2Cyl_DeepCAD", "checkpoint.pth"),
    "with_sketch":    ("Point2Cyl", "model.pth"),
}
_subdir, _fname = _CKPT_PATHS.get(_CKPT_NAME, _CKPT_PATHS["without_sketch"])
WEIGHTS_PATH = os.path.join(THIS_DIR, "results", _subdir, _fname)


def normalize_mesh_points(pts):
    """Center to origin, scale so longest dim of bbox is 1."""
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.max(np.ptp(pts, axis=0))
    if scale < 1e-12:
        scale = 1.0
    pts = pts / scale
    return pts, center, scale


def sample_mesh(mesh_path, n_points, preprocess=False):
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if preprocess:
        # Smooth + decimate to push the noisy scan closer to a clean CAD-like
        # surface (Point2Cyl was trained on sampled CAD meshes).
        try:
            import trimesh.smoothing as ts
            ts.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=20)
        except Exception:
            pass
        try:
            target = max(20000, n_points * 4)
            if len(mesh.faces) > target:
                mesh = mesh.simplify_quadric_decimation(target)
        except Exception:
            pass
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx]
    return np.asarray(pts, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def estimate_axis_for_extrusion(points_norm, point_normals, w_barrel_i, w_base_i):
    """Per-extrusion axis estimation using the BTB - CTC eigen method.

    Reimplementation of Point2Cyl's estimate_extrusion_axis for a single
    extrusion, replacing the deprecated torch.symeig with torch.linalg.eigh.
    """
    # Mask points belonging to this extrusion (soft weights)
    barrel = torch.from_numpy(w_barrel_i).float().unsqueeze(1) * point_normals  # (N, 3)
    base = torch.from_numpy(w_base_i).float().unsqueeze(1) * point_normals
    BTB = barrel.t() @ barrel  # (3, 3)
    CTC = base.t() @ base
    M = BTB - CTC
    # Smallest eigenvalue's eigenvector is the axis
    eigvals, eigvecs = torch.linalg.eigh(M)
    return eigvecs[:, 0].cpu().numpy()


def extrusion_params_from_segment(seg_points, seg_normals, axis):
    """Compute center, extent (along axis), radius of an extrusion segment."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = seg_points.mean(axis=0)
    rel = seg_points - center
    along = rel @ axis
    extent_min, extent_max = float(along.min()), float(along.max())
    extent = extent_max - extent_min
    # Mid-point along axis
    mid_t = 0.5 * (extent_min + extent_max)
    center = center + mid_t * axis
    rel2 = seg_points - center
    perp = rel2 - np.outer(rel2 @ axis, axis)
    radii = np.linalg.norm(perp, axis=1)
    radius = float(np.median(radii))  # robust to outliers
    return center.tolist(), float(extent), radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_path")
    ap.add_argument("output_json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preprocess", action="store_true", help="Smooth + decimate before sampling")
    args = ap.parse_args()

    if not os.path.exists(args.mesh_path):
        print(f"ERROR: mesh not found: {args.mesh_path}", file=sys.stderr)
        sys.exit(2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) Sample point cloud
    pts_world, normals_world = sample_mesh(args.mesh_path, NUM_POINT, preprocess=args.preprocess)
    print(f"  using checkpoint: {_CKPT_NAME} ({WEIGHTS_PATH})")
    # 2) Normalize to unit cube (Point2Cyl trained inputs are normalized)
    pts_norm, center, scale = normalize_mesh_points(pts_world.copy())

    # 3) Load model — DeepCAD checkpoint has an extra center-prediction head
    if _CKPT_NAME == "deepcad":
        output_sizes = [3, 2 * K, 1]  # third head: dummy 1-d (unused at inference)
    else:
        output_sizes = [3, 2 * K]
    model = backbone(output_sizes=output_sizes).to(device).eval()
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    # 4) Forward pass
    pcs = torch.from_numpy(pts_norm).float().unsqueeze(0).to(device)  # (1, N, 3)
    with torch.no_grad():
        outs = model(pcs)
        X, W_raw = outs[0], outs[1]  # ignore center head if present
        X = torch.nn.functional.normalize(X, p=2, dim=2, eps=1e-12)
        W_2K = torch.softmax(W_raw, dim=2)
        W_barrel = W_2K[0, :, ::2].cpu().numpy()  # (N, K)
        W_base = W_2K[0, :, 1::2].cpu().numpy()
        W_seg = W_barrel + W_base                  # (N, K) per-extrusion mass
        pred_normals = X[0].cpu()                  # (N, 3) torch tensor

    # 5) Per-point hard label = argmax over extrusions
    labels = W_seg.argmax(axis=1)  # (N,)

    # 6) For each extrusion that has points, estimate axis + params
    segments_out = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 30:
            continue  # too few points, skip
        seg_pts_norm = pts_norm[idx]
        seg_pts_world = pts_world[idx]
        seg_normals_world = normals_world[idx]

        # Axis from soft weights (uses ALL points so the eigen method has signal)
        axis = estimate_axis_for_extrusion(
            pts_norm, pred_normals, W_barrel[:, k], W_base[:, k]
        )

        center_w, extent_w, radius_w = extrusion_params_from_segment(
            seg_pts_world, seg_normals_world, axis
        )
        # Scale extent + radius back from normalized to world
        # (axis is unitless direction; positions are world-space already)

        segments_out.append({
            "id": int(k),
            "n_points": int(len(idx)),
            "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
            "center": center_w,
            "extent": extent_w,
            "radius": radius_w,
            "points": seg_pts_world.astype(float).tolist(),
        })

    output = {
        "n_segments": len(segments_out),
        "n_points_total": int(NUM_POINT),
        "mesh_center": center.tolist(),
        "mesh_scale": float(scale),
        "segments": segments_out,
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f)
    print(f"OK: wrote {len(segments_out)} extrusion segments to {args.output_json}")


if __name__ == "__main__":
    main()

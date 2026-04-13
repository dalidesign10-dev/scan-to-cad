"""Deviation analysis — compare original mesh against fitted CAD surfaces."""
import numpy as np
import base64
import logging
from .state import PrimitiveType, ConfidenceClass

logger = logging.getLogger(__name__)


def _deviation_to_rgb(deviations, max_dev=None):
    """Map deviations to RGB colors: green(0) -> yellow(mid) -> red(max).

    Returns (N, 3) uint8 array.
    """
    if max_dev is None:
        max_dev = max(np.percentile(deviations, 95), 1e-6)
    t = np.clip(deviations / max_dev, 0, 1)
    # Green -> Yellow -> Red gradient
    r = np.clip(2 * t, 0, 1)
    g = np.clip(2 * (1 - t), 0, 1)
    b = np.zeros_like(t)
    rgb = np.stack([r, g, b], axis=1)
    return (rgb * 255).astype(np.uint8)


def compute_deviation_analysis(state, original_mesh):
    """Compute deviation analysis between original mesh and fitted surfaces.

    Returns dict with stats + per-vertex color data (base64 encoded).
    """
    if state is None or state.snap_result is None:
        raise ValueError("No snap result — run E1 first")

    snap = state.snap_result
    deviations = snap.per_vertex_displacement  # (V,) already computed
    snap_types = snap.vertex_snap_type  # (V,) 0=none, 1=surface, 2=edge, 3=corner
    n_verts = len(deviations)

    # Get reference scale from mesh bounding box
    verts = np.asarray(original_mesh.vertices, dtype=float)
    bbox_diag = float(np.linalg.norm(np.ptp(verts, axis=0)))

    # Stats
    snapped_mask = snap_types > 0
    snapped_devs = deviations[snapped_mask]

    # Tolerance bands (relative to bbox diagonal for scale independence)
    # Also compute absolute bands for display
    tol_bands = [0.001, 0.005, 0.01, 0.02, 0.05]  # fractions of bbox
    abs_bands = [t * bbox_diag for t in tol_bands]
    band_pcts = {}
    for tol, abs_tol in zip(tol_bands, abs_bands):
        pct = float(np.mean(snapped_devs <= abs_tol) * 100) if len(snapped_devs) > 0 else 0
        band_pcts[f"within_{tol*100:.1f}pct_bbox"] = pct
        band_pcts[f"within_{abs_tol:.2f}mm"] = pct

    # Histogram (10 bins)
    if len(snapped_devs) > 0:
        hist_counts, hist_edges = np.histogram(snapped_devs, bins=10)
        histogram = {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        }
    else:
        histogram = {"counts": [], "edges": []}

    # Per-region stats
    region_stats = []
    if state.full_face_region is not None:
        faces = snap.faces
        for rid, region in state.regions.items():
            if region.fit is None:
                continue
            if region.full_face_indices is None or len(region.full_face_indices) == 0:
                continue
            vert_idx = np.unique(faces[region.full_face_indices].ravel())
            region_devs = deviations[vert_idx]
            region_stats.append({
                "region_id": int(rid),
                "type": region.fit.type.value if region.fit.type else "unknown",
                "confidence": region.fit.confidence_class.value if region.fit.confidence_class else "?",
                "mean_dev": float(np.mean(region_devs)),
                "max_dev": float(np.max(region_devs)),
                "n_vertices": int(len(vert_idx)),
            })
        region_stats.sort(key=lambda r: -r["mean_dev"])

    # Color map — per-vertex RGB
    # Use 95th percentile as max for color scaling
    max_dev_color = max(float(np.percentile(deviations, 95)), 1e-6)
    colors = _deviation_to_rgb(deviations, max_dev=max_dev_color)

    # Encode as base64 for transfer (3 bytes per vertex: R,G,B)
    colors_b64 = base64.b64encode(colors.tobytes()).decode("ascii")

    # Per-face colors (average of vertex colors) for the frontend overlay
    # The frontend colors faces, not vertices, so compute per-face deviation
    face_devs = np.mean(deviations[faces], axis=1) if len(faces) > 0 else np.array([])
    face_colors = _deviation_to_rgb(face_devs, max_dev=max_dev_color)
    face_colors_b64 = base64.b64encode(face_colors.tobytes()).decode("ascii")

    return {
        "n_vertices": int(n_verts),
        "n_snapped": int(np.sum(snapped_mask)),
        "pct_snapped": float(np.mean(snapped_mask) * 100),
        "mean_deviation": float(np.mean(deviations)),
        "max_deviation": float(np.max(deviations)),
        "std_deviation": float(np.std(deviations)),
        "median_deviation": float(np.median(deviations)),
        "p95_deviation": float(np.percentile(deviations, 95)),
        "bbox_diagonal": float(bbox_diag),
        "tolerance_bands": band_pcts,
        "histogram": histogram,
        "worst_regions": region_stats[:10],  # top 10 worst
        "best_regions": region_stats[-5:] if len(region_stats) >= 5 else [],
        "vertex_colors_b64": colors_b64,
        "face_colors_b64": face_colors_b64,
        "n_faces": int(len(faces)) if len(faces) > 0 else 0,
        "color_scale_max": float(max_dev_color),
    }

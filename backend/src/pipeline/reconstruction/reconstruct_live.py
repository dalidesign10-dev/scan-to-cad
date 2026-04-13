"""Live region-by-region CAD reconstruction.

Processes each detected region sequentially:
1. AI classifies the region (via Claude API)
2. Builds clean geometry for that region type
3. Checks deviation against original mesh
4. Yields the result as a JSON event for SSE streaming

The frontend renders each surface as it arrives, showing the AI
working in real-time like an engineer in Geomagic.
"""

import json
import math
import logging
import numpy as np
from typing import Generator, Dict, Optional
from scipy.spatial import ConvexHull

from .state import ReconstructionState, Region, PrimitiveType, ConfidenceClass
from .ai_classify import _extract_region_features

logger = logging.getLogger(__name__)


def _build_plane_polygon(vertices, faces, region, params, simplify_tol=0.5):
    """Build a clean flat polygon for a plane region.

    Uses alpha-shape-like approach: project region boundary onto plane,
    simplify with Douglas-Peucker, return clean polygon vertices.
    """
    normal = np.array(params["normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    d = float(params.get("d", 0.0))

    # Get region vertices
    vert_idx = np.unique(faces[region.full_face_indices].ravel())
    pts = vertices[vert_idx]
    if len(pts) < 3:
        return None

    # Project onto plane
    dist = pts @ normal + d
    projected = pts - np.outer(dist, normal)

    # Build 2D frame
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)

    centroid = projected.mean(axis=0)
    rel = projected - centroid
    pts_2d = np.column_stack([rel @ u, rel @ v])

    # Convex hull
    try:
        hull = ConvexHull(pts_2d)
    except Exception:
        return None

    hull_idx = hull.vertices
    hull_2d = pts_2d[hull_idx]

    # Douglas-Peucker simplification
    hull_2d_simplified = _douglas_peucker_closed(hull_2d, simplify_tol)
    if len(hull_2d_simplified) < 3:
        hull_2d_simplified = hull_2d

    # Back to 3D on the plane
    hull_3d = centroid + np.outer(hull_2d_simplified[:, 0], u) + np.outer(hull_2d_simplified[:, 1], v)

    # Fan triangulation
    fan_center = hull_3d.mean(axis=0)
    out_verts = [fan_center.tolist()]
    for p in hull_3d:
        out_verts.append(p.tolist())

    out_faces = []
    n = len(hull_3d)
    for i in range(n):
        out_faces.append([0, 1 + i, 1 + (i + 1) % n])

    return {
        "vertices": out_verts,
        "faces": out_faces,
        "n_polygon_edges": n,
    }


def _build_cylinder_patch(vertices, faces, region, params, n_segments=24):
    """Build a clean cylinder patch for a cylinder region."""
    axis = np.array(params["axis"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-12
    center = np.array(params["center"], dtype=np.float64)
    radius = float(params["radius"])

    vert_idx = np.unique(faces[region.full_face_indices].ravel())
    pts = vertices[vert_idx]
    if len(pts) < 4:
        return None

    # Axial extent
    projections = (pts - center) @ axis
    z_min, z_max = float(projections.min()), float(projections.max())
    height = z_max - z_min
    if height < 1e-6:
        return None

    # Build local frame
    if abs(axis[0]) < 0.9:
        u_ref = np.cross(axis, [1, 0, 0])
    else:
        u_ref = np.cross(axis, [0, 1, 0])
    u_ref /= np.linalg.norm(u_ref) + 1e-12
    v_ref = np.cross(axis, u_ref)

    # Angular extent
    diff = pts - center
    axial = np.outer(diff @ axis, axis)
    radial = diff - axial
    rd = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_dir = radial / np.maximum(rd, 1e-12)
    thetas = np.arctan2(radial_dir @ v_ref, radial_dir @ u_ref)
    theta_min, theta_max = float(thetas.min()), float(thetas.max())

    if theta_max - theta_min > np.pi * 1.5:
        theta_min, theta_max = 0, 2 * np.pi

    # Generate mesh
    theta_range = np.linspace(theta_min, theta_max, n_segments + 1)
    z_range = [z_min, z_max]

    out_verts = []
    for z in z_range:
        for theta in theta_range:
            p = center + z * axis + radius * (np.cos(theta) * u_ref + np.sin(theta) * v_ref)
            out_verts.append(p.tolist())

    out_faces = []
    cols = len(theta_range)
    for col in range(cols - 1):
        i0 = col
        i1 = col + 1
        i2 = cols + col
        i3 = cols + col + 1
        out_faces.append([i0, i2, i1])
        out_faces.append([i1, i2, i3])

    return {
        "vertices": out_verts,
        "faces": out_faces,
        "radius": radius,
        "height": height,
    }


def _compute_region_deviation(original_verts, faces, region, new_verts_3d):
    """Check deviation between original region and new surface."""
    vert_idx = np.unique(faces[region.full_face_indices].ravel())
    original_pts = original_verts[vert_idx]

    if len(original_pts) == 0 or len(new_verts_3d) == 0:
        return {"mean": 0, "max": 0, "pct_within_0_1mm": 0}

    # For each original point, find closest point in new geometry
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array(new_verts_3d))
    dists, _ = tree.query(original_pts)

    return {
        "mean": round(float(np.mean(dists)), 4),
        "max": round(float(np.max(dists)), 4),
        "pct_within_0_1mm": round(float(np.mean(dists < 0.1) * 100), 1),
    }


def _douglas_peucker_closed(pts_2d, tol):
    """Simplify a closed 2D polygon with Douglas-Peucker."""
    n = len(pts_2d)
    if n <= 4:
        return pts_2d

    def dist_pt_seg(p, a, b):
        ab = b - a
        L2 = ab @ ab
        if L2 < 1e-12:
            return float(np.linalg.norm(p - a))
        t = max(0, min(1, ((p - a) @ ab) / L2))
        return float(np.linalg.norm(p - (a + t * ab)))

    # Find the point farthest from the line between first and mid
    keep = np.ones(n, dtype=bool)

    def simplify(start, end):
        if end - start <= 1:
            return
        max_d = 0
        max_i = start
        a, b = pts_2d[start % n], pts_2d[end % n]
        for i in range(start + 1, end):
            d = dist_pt_seg(pts_2d[i % n], a, b)
            if d > max_d:
                max_d = d
                max_i = i
        if max_d > tol:
            simplify(start, max_i)
            simplify(max_i, end)
        else:
            for i in range(start + 1, end):
                keep[i % n] = False

    simplify(0, n)
    return pts_2d[keep]


def _classify_single_region(region, vertices, faces, face_normals, bbox_diag, state, client, api_key):
    """Use Claude to classify a single region."""
    feat = _extract_region_features(region, vertices, faces, face_normals, bbox_diag, state)
    if feat is None:
        return "UNKNOWN", 0.5, "No features"

    prompt = f"""Classify this mesh region as ONE of: PLANE, CYLINDER, CONE, FILLET, CHAMFER, FREEFORM, UNKNOWN.

Region features: {json.dumps(feat)}

Rules:
- normal_spread < 0.05 → PLANE
- normal_spread > 0.1, normal_flatness < 0.1 → CYLINDER
- small area between two planes → FILLET or CHAMFER
- half_angle_deg > 60° labeled as "cone" → probably FILLET

Return ONLY JSON: {{"type": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    import anthropic
    c = client or anthropic.Anthropic(api_key=api_key)
    resp = c.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        result = json.loads(raw)
        return result.get("type", "UNKNOWN"), result.get("confidence", 0.5), result.get("reasoning", "")
    except Exception:
        return "UNKNOWN", 0.3, f"Parse error: {raw[:100]}"


def reconstruct_live(
    state: ReconstructionState,
    original_mesh,
    api_key: str = None,
    use_ai: bool = True,
) -> Generator[dict, None, None]:
    """Generator that yields one event per region as it's reconstructed.

    Each yielded dict has:
      - region_id, step, total
      - classification (AI or geometric)
      - geometry (vertices + faces of the new clean surface)
      - deviation (how well it matches the original)
      - reasoning (AI explanation)

    The frontend can render each surface as it arrives.
    """
    import anthropic

    vertices = np.asarray(original_mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(original_mesh.faces, dtype=np.int64)
    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))

    try:
        face_normals = np.asarray(original_mesh.face_normals, dtype=np.float64)
    except Exception:
        tri = vertices[mesh_faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm = np.linalg.norm(cross, axis=1, keepdims=True)
        face_normals = cross / np.maximum(nrm, 1e-12)

    # Center offset
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

    # Use snapped vertices if available
    work_verts = vertices.copy()
    if state.snap_result is not None:
        work_verts = state.snap_result.snapped_vertices.copy()

    # Sort regions by area (largest first — most impactful)
    sorted_regions = sorted(
        state.regions.values(),
        key=lambda r: r.area_fraction,
        reverse=True,
    )

    # Filter to HIGH confidence only
    high_regions = [
        r for r in sorted_regions
        if r.fit is not None
        and r.fit.confidence_class == ConfidenceClass.HIGH
        and r.full_face_indices is not None
        and len(r.full_face_indices) >= 3
    ]

    total = len(high_regions)
    client = None
    if use_ai and api_key:
        try:
            client = anthropic.Anthropic(api_key=api_key)
        except Exception:
            client = None

    # Accumulated new geometry
    all_new_verts = []
    all_new_faces = []
    vert_offset = 0

    for step, region in enumerate(high_regions):
        event = {
            "region_id": int(region.id),
            "step": step + 1,
            "total": total,
            "area_fraction": round(float(region.area_fraction), 4),
            "status": "processing",
        }

        # 1. Classify
        if use_ai and client:
            try:
                cls_type, confidence, reasoning = _classify_single_region(
                    region, vertices, mesh_faces, face_normals, bbox_diag, state, client, api_key
                )
            except Exception as e:
                cls_type = region.fit.type.value.upper()
                confidence = 0.5
                reasoning = f"AI error: {e}"
        else:
            cls_type = region.fit.type.value.upper()
            confidence = float(region.fit.score) if region.fit.score else 0.8
            reasoning = f"Geometric fit: {region.fit.type.value} rmse={region.fit.rmse:.4f}"

        event["classification"] = cls_type
        event["confidence"] = round(confidence, 2)
        event["reasoning"] = reasoning

        # 2. Build clean geometry based on classification
        geometry = None
        if cls_type == "PLANE":
            geometry = _build_plane_polygon(work_verts, mesh_faces, region, region.fit.params)
        elif cls_type == "CYLINDER":
            geometry = _build_cylinder_patch(work_verts, mesh_faces, region, region.fit.params)
        # FILLET, CHAMFER, CONE, etc. — skip for now (keep as gaps)

        if geometry:
            # Apply center offset
            centered_verts = (np.array(geometry["vertices"]) - center).tolist()
            geometry["vertices"] = centered_verts

            # 3. Check deviation
            deviation = _compute_region_deviation(
                vertices, mesh_faces, region, np.array(centered_verts) + center
            )
            event["deviation"] = deviation
            event["geometry"] = geometry
            event["status"] = "built"

            # Accumulate
            for f in geometry["faces"]:
                all_new_faces.append([f[0] + vert_offset, f[1] + vert_offset, f[2] + vert_offset])
            all_new_verts.extend(geometry["vertices"])
            vert_offset += len(geometry["vertices"])
        else:
            event["status"] = "skipped"
            event["reasoning"] += f" (no geometry builder for {cls_type})"

        yield event

    # Final event with complete mesh
    if all_new_verts:
        yield {
            "status": "complete",
            "total_regions": total,
            "total_built": len(all_new_faces),
            "total_vertices": len(all_new_verts),
        }

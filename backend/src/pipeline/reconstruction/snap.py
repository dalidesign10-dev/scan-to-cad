"""Phase E1 -- vertex snapping to analytic surfaces.

Snaps mesh vertices to the analytic primitives discovered in Phase E0.
Each vertex incident to HIGH-confidence regions is projected onto the
corresponding analytic surface (plane, cylinder, cone), intersection
edge, or corner point. The result is a cleaner mesh that honours the
mechanical intent detected earlier.

Vertex classification:
  0 (NONE)    -- not incident to any HIGH-confidence region.
  1 (SURFACE) -- incident to faces of exactly one HIGH region.
  2 (EDGE)    -- incident to faces of exactly two distinct HIGH regions.
  3 (CORNER)  -- incident to faces of three or more distinct HIGH regions.

All surface projections are vectorised (operate on (N, 3) arrays) so the
common SURFACE case runs in bulk numpy without Python-level loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse

from .state import (
    ConfidenceClass,
    PrimitiveType,
    ReconstructionState,
    Region,
    SurfaceFamily,
)

logger = logging.getLogger(__name__)

# Snap-type codes stored in vertex_snap_type.
SNAP_NONE = np.int8(0)
SNAP_SURFACE = np.int8(1)
SNAP_EDGE = np.int8(2)
SNAP_CORNER = np.int8(3)


# -----------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------

@dataclass
class SnapResult:
    """Output of :func:`snap_mesh_to_analytics`.

    Attributes:
        snapped_vertices: (V, 3) float64 -- vertex positions after snapping.
        faces: (F, 3) int64 -- face indices, unchanged from input.
        vertex_snap_type: (V,) int8 -- per-vertex snap classification
            (0=none, 1=surface, 2=edge, 3=corner).
        per_vertex_displacement: (V,) float64 -- Euclidean distance each
            vertex moved during snapping.
        stats: summary statistics dict.
    """
    snapped_vertices: np.ndarray    # (V, 3) float64
    faces: np.ndarray               # (F, 3) int64 unchanged
    vertex_snap_type: np.ndarray    # (V,) int8: 0=none, 1=surface, 2=edge, 3=corner
    per_vertex_displacement: np.ndarray  # (V,) float64
    stats: dict


# -----------------------------------------------------------------------
# Vectorised surface projections
# -----------------------------------------------------------------------

def _snap_to_plane_batch(
    points: np.ndarray,
    normal: np.ndarray,
    d: float,
) -> np.ndarray:
    """Project points onto the plane ``n . x + d = 0``.

    Args:
        points: (N, 3) float64.
        normal: (3,) unit normal.
        d: signed offset so that ``n . x + d = 0``.

    Returns:
        (N, 3) projected points.
    """
    dist = points @ normal + d  # (N,)
    return points - np.outer(dist, normal)


def _snap_to_cylinder_batch(
    points: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Project points to the surface of an infinite cylinder.

    Args:
        points: (N, 3) float64.
        axis: (3,) unit direction of cylinder axis.
        center: (3,) a point on the axis.
        radius: cylinder radius.

    Returns:
        (N, 3) projected points.
    """
    diff = points - center  # (N, 3)
    axial = np.outer(diff @ axis, axis)  # (N, 3)
    radial = diff - axial  # (N, 3)
    radial_dist = np.linalg.norm(radial, axis=1, keepdims=True)  # (N, 1)
    radial_dist = np.maximum(radial_dist, 1e-12)
    return center + axial + radial * (radius / radial_dist)


def _snap_to_cone_batch(
    points: np.ndarray,
    apex: np.ndarray,
    axis: np.ndarray,
    half_angle_rad: float,
) -> np.ndarray:
    """Project points to the surface of a cone.

    The cone is defined by its apex, a unit axis direction, and a
    half-angle. Points are projected along the direction perpendicular
    to the cone surface.

    Args:
        points: (N, 3) float64.
        apex: (3,) apex of the cone.
        axis: (3,) unit direction from apex along the cone axis.
        half_angle_rad: half-angle in radians.

    Returns:
        (N, 3) projected points.
    """
    diff = points - apex  # (N, 3)
    axial_dist = diff @ axis  # (N,)
    radial_vec = diff - np.outer(axial_dist, axis)  # (N, 3)
    radial_dist = np.linalg.norm(radial_vec, axis=1)  # (N,)
    tan_a = np.tan(half_angle_rad)
    # Optimal projection parameter along the cone generator.
    s = (axial_dist + radial_dist * tan_a) / (1.0 + tan_a ** 2)
    r = s * tan_a
    safe_radial = np.maximum(radial_dist, 1e-12)
    radial_dir = radial_vec / safe_radial[:, None]  # (N, 3) unit radial
    return apex + np.outer(s, axis) + radial_dir * r[:, None]


# -----------------------------------------------------------------------
# Vertex classification
# -----------------------------------------------------------------------

def _build_vertex_face_adjacency(
    n_vertices: int,
    faces: np.ndarray,
) -> sparse.csr_matrix:
    """Build a sparse (V x F) binary matrix: entry (v, f) = 1 iff vertex v
    is used by face f.

    Args:
        n_vertices: total vertex count.
        faces: (F, 3) int64.

    Returns:
        scipy CSR matrix of shape (V, F).
    """
    n_faces = faces.shape[0]
    # Each face contributes 3 entries.
    row = faces.ravel()  # vertex indices
    col = np.repeat(np.arange(n_faces, dtype=np.int64), 3)
    data = np.ones(len(row), dtype=np.int8)
    return sparse.csr_matrix(
        (data, (row, col)),
        shape=(n_vertices, n_faces),
    )


def _classify_vertices(
    full_vertices: np.ndarray,
    full_faces: np.ndarray,
    state: ReconstructionState,
) -> Tuple[np.ndarray, List[Set[int]]]:
    """Classify each vertex by how many distinct HIGH-confidence regions
    it belongs to.

    Args:
        full_vertices: (V, 3).
        full_faces: (F, 3).
        state: reconstruction state with regions and full_face_region.

    Returns:
        vertex_snap_type: (V,) int8 with SNAP_NONE/SURFACE/EDGE/CORNER.
        vertex_high_regions: list of length V, each entry a set of
            region IDs with HIGH fits that the vertex is incident to.
    """
    n_verts = full_vertices.shape[0]
    n_faces = full_faces.shape[0]
    face_region = state.full_face_region  # (F,) int, -1 = unassigned

    # Pre-compute set of HIGH region IDs.
    high_region_ids: Set[int] = set()
    for rid, reg in state.regions.items():
        if (
            reg.fit is not None
            and reg.fit.confidence_class == ConfidenceClass.HIGH
            and not reg.excluded
        ):
            high_region_ids.add(rid)

    # Build face -> is_high mask.
    face_high_region = np.full(n_faces, -1, dtype=np.int64)
    for rid in high_region_ids:
        mask = face_region == rid
        face_high_region[mask] = rid

    # Build sparse vertex-face adjacency.
    vf = _build_vertex_face_adjacency(n_verts, full_faces)

    # For each vertex collect incident HIGH region IDs.
    vertex_high_regions: List[Set[int]] = [set() for _ in range(n_verts)]
    vertex_snap_type = np.full(n_verts, SNAP_NONE, dtype=np.int8)

    # Iterate through vertices that have at least one incident face.
    for vi in range(n_verts):
        start, end = vf.indptr[vi], vf.indptr[vi + 1]
        if start == end:
            continue
        face_indices = vf.indices[start:end]
        region_ids_for_faces = face_high_region[face_indices]
        unique_high = set(region_ids_for_faces[region_ids_for_faces >= 0].tolist())
        vertex_high_regions[vi] = unique_high
        n_high = len(unique_high)
        if n_high == 0:
            vertex_snap_type[vi] = SNAP_NONE
        elif n_high == 1:
            vertex_snap_type[vi] = SNAP_SURFACE
        elif n_high == 2:
            vertex_snap_type[vi] = SNAP_EDGE
        else:
            vertex_snap_type[vi] = SNAP_CORNER

    return vertex_snap_type, vertex_high_regions


# -----------------------------------------------------------------------
# Edge snap helpers
# -----------------------------------------------------------------------

def _snap_to_line(
    point: np.ndarray,
    line_point: np.ndarray,
    line_dir: np.ndarray,
) -> np.ndarray:
    """Project a single point onto an infinite line.

    Args:
        point: (3,) query point.
        line_point: (3,) any point on the line.
        line_dir: (3,) unit direction of the line.

    Returns:
        (3,) closest point on the line.
    """
    diff = point - line_point
    t = float(np.dot(diff, line_dir))
    return line_point + t * line_dir


def _snap_to_polyline(
    point: np.ndarray,
    polyline_points: np.ndarray,
) -> np.ndarray:
    """Find the nearest point on a sampled polyline.

    For each consecutive segment of the polyline, project the query point
    onto the segment and keep the closest projection.

    Args:
        point: (3,) query point.
        polyline_points: (M, 3) ordered polyline vertices, M >= 2.

    Returns:
        (3,) nearest point on the polyline.
    """
    if polyline_points.shape[0] < 2:
        return polyline_points[0].copy()

    best_dist_sq = np.inf
    best_pt = polyline_points[0].copy()

    segments_start = polyline_points[:-1]  # (M-1, 3)
    segments_end = polyline_points[1:]     # (M-1, 3)
    seg_vec = segments_end - segments_start  # (M-1, 3)
    seg_len_sq = np.sum(seg_vec ** 2, axis=1)  # (M-1,)

    diff = point - segments_start  # (M-1, 3)
    t = np.sum(diff * seg_vec, axis=1)  # (M-1,)
    # Clamp t to [0, seg_len_sq] -- safe div follows.
    safe_len_sq = np.maximum(seg_len_sq, 1e-24)
    t_norm = np.clip(t / safe_len_sq, 0.0, 1.0)  # (M-1,)
    proj = segments_start + t_norm[:, None] * seg_vec  # (M-1, 3)
    dist_sq = np.sum((point - proj) ** 2, axis=1)  # (M-1,)

    idx = int(np.argmin(dist_sq))
    return proj[idx]


# -----------------------------------------------------------------------
# Corner snap
# -----------------------------------------------------------------------

def _snap_to_corner(
    point: np.ndarray,
    families: List[SurfaceFamily],
    max_iter: int = 20,
    tol: float = 1e-10,
) -> np.ndarray:
    """Snap a vertex to the intersection of 3+ analytic surfaces.

    For all-plane corners the system is solved directly via least-squares.
    For mixed surface types an iterative alternating-projection scheme is
    used: project onto each surface in turn, average, repeat.

    Args:
        point: (3,) starting point.
        families: list of SurfaceFamily objects (len >= 3).
        max_iter: iteration limit for alternating projection.
        tol: convergence tolerance on displacement.

    Returns:
        (3,) snapped corner position.
    """
    # Check if all planes -- direct linear solve.
    all_planes = all(f.type == PrimitiveType.PLANE for f in families)

    if all_planes:
        # Each plane gives n . x + d = 0  -->  n . x = -d
        normals = []
        offsets = []
        for f in families:
            n = np.asarray(f.canonical_params["normal"], dtype=np.float64)
            d = float(f.canonical_params["d"])
            normals.append(n)
            offsets.append(-d)
        A = np.array(normals)  # (K, 3)
        b = np.array(offsets)  # (K,)
        # Least-squares solve (handles over-determined case).
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result

    # Mixed surfaces: iterative alternating projection.
    current = point.copy()
    for _ in range(max_iter):
        projections = []
        for f in families:
            p = current[None, :]  # (1, 3)
            if f.type == PrimitiveType.PLANE:
                n = np.asarray(f.canonical_params["normal"], dtype=np.float64)
                d = float(f.canonical_params["d"])
                proj = _snap_to_plane_batch(p, n, d)
            elif f.type == PrimitiveType.CYLINDER:
                ax = np.asarray(f.canonical_params["axis"], dtype=np.float64)
                ctr = np.asarray(f.canonical_params["center"], dtype=np.float64)
                rad = float(f.canonical_params["radius"])
                proj = _snap_to_cylinder_batch(p, ax, ctr, rad)
            elif f.type == PrimitiveType.CONE:
                apex = np.asarray(f.canonical_params["apex"], dtype=np.float64)
                ax = np.asarray(f.canonical_params["axis"], dtype=np.float64)
                ha_deg = float(f.canonical_params["half_angle_deg"])
                ha_rad = np.radians(ha_deg)
                proj = _snap_to_cone_batch(p, apex, ax, ha_rad)
            else:
                # Unknown primitive -- skip.
                continue
            projections.append(proj[0])

        if not projections:
            break
        avg = np.mean(projections, axis=0)
        displacement = float(np.linalg.norm(avg - current))
        current = avg
        if displacement < tol:
            break

    return current


# -----------------------------------------------------------------------
# Internal helpers for the main orchestrator
# -----------------------------------------------------------------------

def _get_region_rmse(region: Region) -> float:
    """Return the RMSE from a region's fit, or inf if unavailable."""
    if region.fit is None:
        return np.inf
    return float(region.fit.rmse)


def _snap_surface_vertices(
    vertices: np.ndarray,
    vertex_indices: np.ndarray,
    region: Region,
) -> np.ndarray:
    """Batch-snap a set of SURFACE-classified vertices to a single region's
    analytic primitive.

    Args:
        vertices: (V, 3) full vertex array (read-only reference).
        vertex_indices: (K,) indices of vertices to snap.
        region: the region whose fit defines the target surface.

    Returns:
        (K, 3) snapped positions.
    """
    pts = vertices[vertex_indices]  # (K, 3)
    fit = region.fit
    if fit is None or fit.type == PrimitiveType.UNKNOWN:
        return pts.copy()

    params = fit.params
    if fit.type == PrimitiveType.PLANE:
        normal = np.asarray(params["normal"], dtype=np.float64)
        d = float(params["d"])
        return _snap_to_plane_batch(pts, normal, d)

    if fit.type == PrimitiveType.CYLINDER:
        axis = np.asarray(params["axis"], dtype=np.float64)
        center = np.asarray(params["center"], dtype=np.float64)
        radius = float(params["radius"])
        return _snap_to_cylinder_batch(pts, axis, center, radius)

    if fit.type == PrimitiveType.CONE:
        apex = np.asarray(params["apex"], dtype=np.float64)
        axis = np.asarray(params["axis"], dtype=np.float64)
        half_angle_deg = float(params["half_angle_deg"])
        half_angle_rad = np.radians(half_angle_deg)
        return _snap_to_cone_batch(pts, apex, axis, half_angle_rad)

    return pts.copy()


def _build_edge_lookup(
    state: ReconstructionState,
) -> Dict[Tuple[int, int], dict]:
    """Build a lookup from (family_a, family_b) -> intent_edge dict.

    Keys are sorted so that family_a < family_b.

    Args:
        state: reconstruction state with intent_edges populated.

    Returns:
        dict mapping (fa, fb) -> edge dict with "points" polyline.
    """
    lookup: Dict[Tuple[int, int], dict] = {}
    for edge in state.intent_edges:
        fa = int(edge.get("family_a", -1))
        fb = int(edge.get("family_b", -1))
        if fa < 0 or fb < 0:
            continue
        key = (fa, fb) if fa < fb else (fb, fa)
        # If duplicates exist, keep the one with more points.
        if key in lookup:
            existing_n = lookup[key].get("n_points", 0)
            new_n = edge.get("n_points", 0)
            if new_n <= existing_n:
                continue
        lookup[key] = edge
    return lookup


def _region_to_family(
    region_id: int,
    state: ReconstructionState,
) -> int:
    """Return the surface_family_id for a region, or -1."""
    region = state.regions.get(region_id)
    if region is None:
        return -1
    return region.surface_family_id


def _snap_edge_vertex(
    point: np.ndarray,
    region_ids: Set[int],
    state: ReconstructionState,
    edge_lookup: Dict[Tuple[int, int], dict],
) -> np.ndarray:
    """Snap a single EDGE-classified vertex.

    If an intent_edge polyline exists for the family pair, project onto
    that polyline. Otherwise fall back to alternating projection between
    the two surfaces.

    Args:
        point: (3,) vertex position.
        region_ids: exactly 2 region IDs with HIGH fits.
        state: reconstruction state.
        edge_lookup: family-pair -> edge dict lookup.

    Returns:
        (3,) snapped position.
    """
    rids = sorted(region_ids)
    fa = _region_to_family(rids[0], state)
    fb = _region_to_family(rids[1], state)
    if fa < 0 or fb < 0:
        return point.copy()

    key = (fa, fb) if fa < fb else (fb, fa)
    edge = edge_lookup.get(key)

    if edge is not None:
        pts_raw = edge.get("points")
        if pts_raw is not None and len(pts_raw) >= 2:
            polyline = np.asarray(pts_raw, dtype=np.float64)
            if polyline.ndim == 2 and polyline.shape[0] >= 2:
                # For a 2-point polyline (line segment from plane/plane),
                # use infinite line projection for better accuracy.
                if polyline.shape[0] == 2:
                    line_dir = polyline[1] - polyline[0]
                    length = float(np.linalg.norm(line_dir))
                    if length > 1e-12:
                        line_dir /= length
                        return _snap_to_line(point, polyline[0], line_dir)
                return _snap_to_polyline(point, polyline)

    # Fallback: alternating projection between the two surfaces.
    families = []
    for fid in (fa, fb):
        fam = state.surface_families.get(fid)
        if fam is not None:
            families.append(fam)
    if len(families) == 2:
        return _snap_to_corner(point, families, max_iter=30, tol=1e-10)

    return point.copy()


# -----------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------

def snap_mesh_to_analytics(
    state: ReconstructionState,
    full_mesh,
) -> SnapResult:
    """Snap all HIGH-region vertices to their analytic surfaces.

    This is the main entry point for Phase E1 vertex snapping. Each
    vertex incident to HIGH-confidence regions is projected onto the
    closest analytic primitive, intersection edge, or corner.

    Args:
        state: ReconstructionState with regions, surface_families,
            intent_edges, and full_face_region populated.
        full_mesh: trimesh.Trimesh -- the full-resolution cleaned mesh.

    Returns:
        SnapResult with snapped vertex positions and diagnostics.

    Raises:
        ValueError: if state.full_face_region is None.
    """
    # ---- Validate inputs ------------------------------------------------
    if state.full_face_region is None:
        raise ValueError(
            "state.full_face_region must be populated before snapping. "
            "Run region assignment first."
        )

    vertices = np.asarray(full_mesh.vertices, dtype=np.float64).copy()  # (V, 3)
    faces = np.asarray(full_mesh.faces, dtype=np.int64)  # (F, 3)
    n_verts = vertices.shape[0]

    if n_verts == 0:
        return SnapResult(
            snapped_vertices=vertices,
            faces=faces,
            vertex_snap_type=np.zeros(0, dtype=np.int8),
            per_vertex_displacement=np.zeros(0, dtype=np.float64),
            stats={"n_snapped": 0, "n_edge": 0, "n_corner": 0,
                   "pct_snapped": 0.0, "rmse_before": 0.0, "rmse_after": 0.0},
        )

    # ---- Reference scale from bounding box ------------------------------
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    reference_scale = float(np.linalg.norm(bbox_max - bbox_min))
    if reference_scale < 1e-15:
        reference_scale = 1.0

    # ---- Step 1: classify vertices --------------------------------------
    vertex_snap_type, vertex_high_regions = _classify_vertices(
        vertices, faces, state,
    )

    # ---- Step 2: build edge lookup from intent_edges --------------------
    edge_lookup = _build_edge_lookup(state)

    # ---- Step 3: batch snap SURFACE vertices ----------------------------
    snapped = vertices.copy()  # working copy
    surface_mask = vertex_snap_type == SNAP_SURFACE

    # Group surface vertices by region for batch projection.
    region_to_verts: Dict[int, List[int]] = {}
    surface_indices = np.where(surface_mask)[0]
    for vi in surface_indices:
        rids = vertex_high_regions[vi]
        if len(rids) != 1:
            continue
        rid = next(iter(rids))
        region_to_verts.setdefault(rid, []).append(vi)

    for rid, vert_list in region_to_verts.items():
        region = state.regions.get(rid)
        if region is None or region.fit is None:
            continue
        vi_arr = np.array(vert_list, dtype=np.int64)
        new_pos = _snap_surface_vertices(vertices, vi_arr, region)
        snapped[vi_arr] = new_pos

    # ---- Step 4: per-vertex snap EDGE vertices --------------------------
    edge_mask = vertex_snap_type == SNAP_EDGE
    edge_indices = np.where(edge_mask)[0]
    for vi in edge_indices:
        rids = vertex_high_regions[vi]
        if len(rids) != 2:
            continue
        snapped[vi] = _snap_edge_vertex(
            vertices[vi], rids, state, edge_lookup,
        )

    # ---- Step 5: per-vertex snap CORNER vertices ------------------------
    corner_mask = vertex_snap_type == SNAP_CORNER
    corner_indices = np.where(corner_mask)[0]
    for vi in corner_indices:
        rids = vertex_high_regions[vi]
        if len(rids) < 3:
            continue
        families = []
        seen_fam_ids: Set[int] = set()
        for rid in rids:
            fid = _region_to_family(rid, state)
            if fid < 0 or fid in seen_fam_ids:
                continue
            fam = state.surface_families.get(fid)
            if fam is not None:
                families.append(fam)
                seen_fam_ids.add(fid)
        if len(families) >= 3:
            snapped[vi] = _snap_to_corner(vertices[vi], families)
        elif len(families) == 2:
            # Degenerate corner: only two distinct families. Treat as edge.
            snapped[vi] = _snap_to_corner(
                vertices[vi], families, max_iter=30, tol=1e-10,
            )

    # ---- Step 5b: snap NONE vertices to nearest HIGH region ---------------
    # Vertices not in any HIGH region keep scan noise. Find the nearest
    # HIGH region for each and snap to its surface.
    none_mask = vertex_snap_type == SNAP_NONE
    none_indices = np.where(none_mask)[0]
    if len(none_indices) > 0 and state.full_face_region is not None:
        # Build a KD-tree of HIGH region centroids for fast nearest lookup
        from scipy.spatial import cKDTree
        high_rids = [rid for rid, r in state.regions.items()
                     if r.fit and r.fit.confidence_class == ConfidenceClass.HIGH]
        if high_rids:
            # Compute centroid of each HIGH region
            centroids = []
            for rid in high_rids:
                r = state.regions[rid]
                vidx = np.unique(faces[r.full_face_indices].ravel())
                centroids.append(vertices[vidx].mean(axis=0))
            centroids = np.array(centroids)
            tree = cKDTree(centroids)

            # For each NONE vertex, find nearest HIGH region and snap
            none_pts = vertices[none_indices]
            _, nearest_idx = tree.query(none_pts)
            for i, vi in enumerate(none_indices):
                rid = high_rids[nearest_idx[i]]
                region = state.regions[rid]
                new_pos = _snap_surface_vertices(
                    vertices, np.array([vi], dtype=np.int64), region
                )
                snapped[vi] = new_pos[0]

    # ---- Step 6: clamp wild displacements (safety net) -------------------
    displacement = np.linalg.norm(snapped - vertices, axis=1)  # (V,)
    # Clamp vertices that moved more than 1% of bbox — these are bad snaps
    wild_threshold = 0.005 * reference_scale
    wild_mask = displacement > wild_threshold
    if np.any(wild_mask):
        wild_idx = np.where(wild_mask)[0]
        for vi in wild_idx:
            snapped[vi] = vertices[vi]  # revert wild vertices to original
        displacement = np.linalg.norm(snapped - vertices, axis=1)

    # ---- Step 7: compute stats ------------------------------------------
    snapped_mask = vertex_snap_type > SNAP_NONE
    n_snapped = int(np.count_nonzero(snapped_mask))
    n_edge = int(np.count_nonzero(edge_mask))
    n_corner = int(np.count_nonzero(corner_mask))
    n_surface = int(np.count_nonzero(surface_mask))
    pct_snapped = 100.0 * n_snapped / max(n_verts, 1)

    # RMSE of displacement for snapped vertices.
    if n_snapped > 0:
        rmse_displacement = float(
            np.sqrt(np.mean(displacement[snapped_mask] ** 2))
        )
        mean_displacement = float(np.mean(displacement[snapped_mask]))
        max_displacement = float(np.max(displacement[snapped_mask]))
    else:
        rmse_displacement = 0.0
        mean_displacement = 0.0
        max_displacement = 0.0

    stats = {
        "n_vertices": int(n_verts),
        "n_snapped": n_snapped,
        "n_surface": n_surface,
        "n_edge": n_edge,
        "n_corner": n_corner,
        "pct_snapped": round(pct_snapped, 2),
        "rmse_displacement": rmse_displacement,
        "mean_displacement": mean_displacement,
        "max_displacement": max_displacement,
        "reference_scale": reference_scale,
    }

    logger.info(
        "Vertex snap complete: %d/%d vertices snapped (%.1f%%), "
        "%d surface, %d edge, %d corner, RMSE displacement %.6f",
        n_snapped, n_verts, pct_snapped,
        n_surface, n_edge, n_corner, rmse_displacement,
    )

    return SnapResult(
        snapped_vertices=snapped,
        faces=faces,
        vertex_snap_type=vertex_snap_type,
        per_vertex_displacement=displacement,
        stats=stats,
    )

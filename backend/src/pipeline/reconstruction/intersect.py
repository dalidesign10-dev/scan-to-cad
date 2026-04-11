"""Phase E0 — family-level analytic intersections.

Every SurfaceFamily is a canonical analytic surface; two surfaces that
meet along a sharp edge of the part imply a real analytic curve (a
line for plane/plane, an ellipse for plane/cylinder, etc.). This
module produces that set of curves.

Design rules:

  1. Family-level, not region-level. Two coplanar pads share a family,
     so they never produce a duplicate intersection — one plane pair
     yields one line, regardless of how many physically separate
     regions sit on it.

  2. Adjacency-pruned. We do NOT compute every O(n²) family pair; we
     walk ``state.boundaries`` and only intersect family pairs whose
     member regions actually share a SHARP boundary. That trims a
     ~3600-pair phantom set on a real scan down to the ~50 edges the
     user can see on the mesh.

  3. JSON-shaped output. Each intersection is a plain dict matching
     the frontend's IntentFamilyEdge interface — keeps the overlay
     shipping path dead simple.

  4. Plane/plane only for now. Plane/cylinder, plane/cone and
     cylinder/cylinder are tractable but each needs their own trimming
     story (sampled curves + KDTree filter against region point
     clouds) so they belong to a follow-up pass. The skeleton here
     returns early on non-plane pairs rather than producing garbage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .state import (
    Boundary,
    PrimitiveType,
    Region,
    ReconstructionState,
    SurfaceFamily,
)


# ─────────────────────────────────────────────────────────────────────────
# Clipping helpers
# ─────────────────────────────────────────────────────────────────────────

def _clip_line_to_aabb(
    p0: np.ndarray,
    direction: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Liang-Barsky line-vs-AABB clip.

    ``p0`` is any point on the infinite line, ``direction`` is a unit
    vector. Returns the two clipped endpoints, or None if the line
    misses the box (or is degenerate in every axis and out of range).
    """
    t_min, t_max = -np.inf, np.inf
    for ax in range(3):
        if abs(direction[ax]) < 1e-9:
            if p0[ax] < bbox_min[ax] or p0[ax] > bbox_max[ax]:
                return None
            continue
        t1 = (bbox_min[ax] - p0[ax]) / direction[ax]
        t2 = (bbox_max[ax] - p0[ax]) / direction[ax]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
        if t2 < t_max:
            t_max = t2
        if t_min > t_max:
            return None
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return None
    return p0 + t_min * direction, p0 + t_max * direction


# ─────────────────────────────────────────────────────────────────────────
# Family pair discovery
# ─────────────────────────────────────────────────────────────────────────

def _adjacent_family_pairs(
    state: ReconstructionState,
) -> Dict[Tuple[int, int], List[int]]:
    """Walk state.boundaries and collect family pairs that share at least
    one sharp boundary.

    Returns a dict keyed by ``(fa, fb)`` with ``fa < fb``, whose values
    are the boundary indices that map to that family pair (handy for
    downstream provenance / debugging).

    Two regions on the SAME family (coplanar pads) are skipped — there
    is no intersection to compute, they are literally the same surface.
    Non-HIGH regions (family_id < 0) are skipped because they have no
    canonical primitive.
    """
    pairs: Dict[Tuple[int, int], List[int]] = {}
    for bi, b in enumerate(state.boundaries):
        if not b.sharp:
            continue
        ra = state.regions.get(b.region_a)
        rb = state.regions.get(b.region_b)
        if ra is None or rb is None:
            continue
        fa = ra.surface_family_id
        fb = rb.surface_family_id
        if fa < 0 or fb < 0:
            continue
        # Both regions must live in a SurfaceFamily object — that
        # guarantees HIGH + known primitive type.
        if fa not in state.surface_families or fb not in state.surface_families:
            continue
        if fa == fb:
            continue
        key = (fa, fb) if fa < fb else (fb, fa)
        pairs.setdefault(key, []).append(bi)
    return pairs


# ─────────────────────────────────────────────────────────────────────────
# Individual intersection cases
# ─────────────────────────────────────────────────────────────────────────

def _plane_plane_edge(
    fa: SurfaceFamily,
    fb: SurfaceFamily,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Optional[List[List[float]]]:
    """Plane × plane → line segment, clipped to the mesh AABB."""
    pa = fa.canonical_params
    pb = fb.canonical_params
    try:
        n1 = np.asarray(pa["normal"], dtype=float)
        n2 = np.asarray(pb["normal"], dtype=float)
        d1 = float(pa["d"])
        d2 = float(pb["d"])
    except (KeyError, TypeError, ValueError):
        return None
    # Parallel check: |n1 × n2| ≈ 0 means parallel planes.
    direction = np.cross(n1, n2)
    denom = float(np.linalg.norm(direction))
    if denom < 1e-6:
        return None
    direction /= denom
    # Point on the line: solve n1·p = -d1, n2·p = -d2, direction·p = 0.
    # (The direction·p = 0 row picks the foot of the perpendicular from
    # the origin onto the line, which keeps p0 numerically stable.)
    A = np.stack([n1, n2, direction])
    b = np.array([-d1, -d2, 0.0])
    try:
        p0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    clipped = _clip_line_to_aabb(p0, direction, bbox_min, bbox_max)
    if clipped is None:
        return None
    a, c = clipped
    return [a.tolist(), c.tolist()]


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def compute_family_edges(
    state: ReconstructionState,
    mesh_vertices: Optional[np.ndarray] = None,
) -> List[dict]:
    """Build the family-level intersection edges for a ReconstructionState.

    Requires ``state.surface_families`` to be populated. Supply
    ``mesh_vertices`` when available — we use them to derive the AABB
    used for plane-plane clipping. If absent we fall back to a bbox
    derived from the proxy mesh (always available).

    Returns a list of JSON-shaped edge dicts. Current schema (subject
    to extension, not removal):

        {
          "family_a": int,
          "family_b": int,
          "type_a":   "plane" | ...,
          "type_b":   "plane" | ...,
          "kind":     "plane_plane",
          "points":   [[x,y,z], [x,y,z], ...],
          "n_points": int,
          "n_supporting_boundaries": int,   # for provenance
        }

    The ``points`` list is a polyline (length ≥ 2). For plane/plane it
    is exactly the two clipped line endpoints.
    """
    if not state.surface_families:
        return []

    if mesh_vertices is not None and mesh_vertices.size > 0:
        verts = np.asarray(mesh_vertices, dtype=float)
    else:
        verts = np.asarray(state.proxy.vertices, dtype=float)
    if verts.size == 0:
        return []
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    pad = 0.02 * diag
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad

    pairs = _adjacent_family_pairs(state)
    if not pairs:
        return []

    edges: List[dict] = []
    for (fa_id, fb_id), boundary_ids in pairs.items():
        fa = state.surface_families[fa_id]
        fb = state.surface_families[fb_id]

        # Only plane/plane in this pass. Everything else is silently
        # skipped — callers will see the short edge list and know to
        # add the other cases in follow-ups rather than chasing a bug.
        if fa.type != PrimitiveType.PLANE or fb.type != PrimitiveType.PLANE:
            continue

        points = _plane_plane_edge(fa, fb, bbox_min, bbox_max)
        if points is None:
            continue

        edges.append({
            "family_a": int(fa_id),
            "family_b": int(fb_id),
            "type_a": fa.type.value,
            "type_b": fb.type.value,
            "kind": "plane_plane",
            "points": points,
            "n_points": len(points),
            "n_supporting_boundaries": len(boundary_ids),
        })

    return edges

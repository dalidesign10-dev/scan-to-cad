"""Phase E2 -- Trimmed face construction.

Builds closed boundary loops around each HIGH-confidence analytic region,
parameterizes them in surface-local UV space, and packages them as
TrimmedFace objects ready for downstream BREP export or CAD kernel
ingestion.

Entry point: :func:`construct_trimmed_faces`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .state import (
    BoundaryLoop,
    ConfidenceClass,
    PrimitiveType,
    ReconstructionState,
    TrimmedFace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boundary edge extraction
# ---------------------------------------------------------------------------

def _extract_boundary_edges(
    face_indices: np.ndarray,
    mesh_faces: np.ndarray,
) -> List[Tuple[int, int]]:
    """Extract boundary edges of a region.

    A boundary edge is one where exactly one of its two adjacent faces
    belongs to the region.

    Args:
        face_indices: (K,) int array of face indices belonging to this region.
        mesh_faces: (F, 3) int array of all mesh faces.

    Returns:
        List of (v_from, v_to) directed edge pairs forming the boundary.
    """
    region_faces = mesh_faces[face_indices]  # (K, 3)

    # Build all directed half-edges: for face [a, b, c] -> (a,b), (b,c), (c,a)
    e0 = region_faces[:, [0, 1]]
    e1 = region_faces[:, [1, 2]]
    e2 = region_faces[:, [2, 0]]
    half_edges = np.concatenate([e0, e1, e2], axis=0)  # (3K, 2)

    # A half-edge is interior if its reverse also appears in the set.
    # Encode edges as single int64 for fast set membership.
    max_v = int(half_edges.max()) + 1
    forward_keys = half_edges[:, 0].astype(np.int64) * max_v + half_edges[:, 1].astype(np.int64)
    reverse_keys = half_edges[:, 1].astype(np.int64) * max_v + half_edges[:, 0].astype(np.int64)

    forward_set = set(forward_keys.tolist())
    boundary_mask = np.array(
        [rk not in forward_set for rk in reverse_keys.tolist()],
        dtype=bool,
    )

    boundary = half_edges[boundary_mask]
    return [(int(a), int(b)) for a, b in boundary]


# ---------------------------------------------------------------------------
# Boundary loop ordering
# ---------------------------------------------------------------------------

def _order_boundary_loops(
    boundary_edges: List[Tuple[int, int]],
) -> List[List[int]]:
    """Order boundary edges into closed loops.

    A region may have multiple loops (outer boundary + holes).

    Args:
        boundary_edges: list of (v_from, v_to) tuples.

    Returns:
        List of loops, each loop is a list of vertex indices in order.
        The longest loop appears first (typically the outer boundary);
        shorter loops follow (holes / inner boundaries).
    """
    if not boundary_edges:
        return []

    # Adjacency: vertex -> list of successor vertices
    adj: Dict[int, List[int]] = defaultdict(list)
    for v_from, v_to in boundary_edges:
        adj[v_from].append(v_to)

    visited_edges: set = set()
    loops: List[List[int]] = []

    for start_from, start_to in boundary_edges:
        edge_key = (start_from, start_to)
        if edge_key in visited_edges:
            continue

        loop: List[int] = [start_from]
        current = start_from
        next_v = start_to

        while True:
            visited_edges.add((current, next_v))
            loop.append(next_v)

            if next_v == start_from:
                # Closed the loop -- drop the duplicated start vertex at end
                loop.pop()
                break

            # Walk forward: pick an unvisited successor
            successors = adj.get(next_v, [])
            advanced = False
            for s in successors:
                if (next_v, s) not in visited_edges:
                    current = next_v
                    next_v = s
                    advanced = True
                    break

            if not advanced:
                # Dangling chain -- should not happen on a valid manifold
                logger.warning(
                    "Boundary loop walk stuck at vertex %d with %d edges visited; "
                    "dropping incomplete loop of length %d.",
                    next_v, len(visited_edges), len(loop),
                )
                loop = []
                break

        if loop:
            loops.append(loop)

    # Sort: longest first (outer), then descending length
    loops.sort(key=lambda lp: len(lp), reverse=True)
    return loops


# ---------------------------------------------------------------------------
# UV parameterization
# ---------------------------------------------------------------------------

def _parameterize_plane(
    points_3d: np.ndarray,
    normal: np.ndarray,
    d: float,
) -> np.ndarray:
    """Compute UV coordinates for points on a plane.

    Builds a local orthonormal frame from the plane normal and projects
    each point onto the two tangent axes.

    Args:
        points_3d: (N, 3) world-space positions.
        normal: (3,) unit normal of the plane.
        d: signed distance from origin (plane eq: normal . x = d).

    Returns:
        (N, 2) array of (u, v) coordinates.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-30)

    # Choose a reference vector not parallel to the normal
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u_axis = np.cross(normal, ref)
    u_axis /= np.linalg.norm(u_axis) + 1e-30
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis) + 1e-30

    # Project onto the plane origin (closest point to world origin)
    origin = normal * d
    rel = points_3d - origin  # (N, 3)

    u = rel @ u_axis
    v = rel @ v_axis
    return np.column_stack([u, v])


def _parameterize_cylinder(
    points_3d: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Compute UV coordinates for points on a cylinder.

    u = theta (angle around axis, in radians), v = axial position.

    Args:
        points_3d: (N, 3) world-space positions.
        axis: (3,) unit direction of cylinder axis.
        center: (3,) a point on the axis.
        radius: cylinder radius.

    Returns:
        (N, 2) array of (theta, axial_position).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    center = np.asarray(center, dtype=np.float64)

    rel = points_3d - center  # (N, 3)

    # Axial component
    v = rel @ axis  # (N,)

    # Radial component: subtract axial projection
    axial_proj = np.outer(v, axis)  # (N, 3)
    radial = rel - axial_proj  # (N, 3)

    # Build consistent local frame for theta
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(axis, ref)
    x_axis /= np.linalg.norm(x_axis) + 1e-30
    y_axis = np.cross(axis, x_axis)

    u = np.arctan2(radial @ y_axis, radial @ x_axis)  # (N,)
    return np.column_stack([u, v])


def _parameterize_cone(
    points_3d: np.ndarray,
    apex: np.ndarray,
    axis: np.ndarray,
    half_angle_rad: float,
) -> np.ndarray:
    """Compute UV coordinates for points on a cone.

    u = theta (angle around axis, in radians),
    v = distance from apex projected onto the axis.

    Args:
        points_3d: (N, 3) world-space positions.
        apex: (3,) apex of the cone.
        axis: (3,) unit direction from apex toward the opening.
        half_angle_rad: half-angle of the cone in radians.

    Returns:
        (N, 2) array of (theta, axial_distance).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    apex = np.asarray(apex, dtype=np.float64)

    rel = points_3d - apex  # (N, 3)

    # Axial distance from apex
    v = rel @ axis  # (N,)

    # Radial component
    axial_proj = np.outer(v, axis)
    radial = rel - axial_proj

    # Build consistent local frame for theta
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(axis, ref)
    x_axis /= np.linalg.norm(x_axis) + 1e-30
    y_axis = np.cross(axis, x_axis)

    u = np.arctan2(radial @ y_axis, radial @ x_axis)  # (N,)
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _loop_area_3d(vertices_3d: np.ndarray) -> float:
    """Compute the approximate area enclosed by a 3D polygon loop.

    Uses the shoelace formula projected onto the best-fit plane via the
    Newell method for the polygon normal.

    Args:
        vertices_3d: (N, 3) ordered 3D vertices of a closed polygon.

    Returns:
        Scalar area (always non-negative).
    """
    n = len(vertices_3d)
    if n < 3:
        return 0.0

    # Newell's method for polygon normal -- also gives 2 * area as
    # the magnitude of the cross-product sum.
    cross_sum = np.zeros(3, dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        cross_sum += np.cross(vertices_3d[i], vertices_3d[j])

    return 0.5 * float(np.linalg.norm(cross_sum))


def _compute_uv(
    points_3d: np.ndarray,
    surface_type: PrimitiveType,
    params: dict,
) -> Optional[np.ndarray]:
    """Dispatch UV parameterization based on surface type.

    Returns (N, 2) UV array, or None if parameterization is unavailable.
    """
    try:
        if surface_type == PrimitiveType.PLANE:
            normal = np.asarray(params["normal"], dtype=np.float64)
            d = float(params["d"])
            return _parameterize_plane(points_3d, normal, d)

        elif surface_type == PrimitiveType.CYLINDER:
            axis = np.asarray(params["axis"], dtype=np.float64)
            center = np.asarray(params["center"], dtype=np.float64)
            radius = float(params["radius"])
            return _parameterize_cylinder(points_3d, axis, center, radius)

        elif surface_type == PrimitiveType.CONE:
            apex = np.asarray(params["apex"], dtype=np.float64)
            axis = np.asarray(params["axis"], dtype=np.float64)
            half_angle = float(params["half_angle_rad"])
            return _parameterize_cone(points_3d, apex, axis, half_angle)

    except (KeyError, TypeError) as exc:
        logger.warning("UV parameterization failed for %s: %s", surface_type, exc)
        return None

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct_trimmed_faces(
    state: ReconstructionState,
    full_mesh,
) -> Dict[int, TrimmedFace]:
    """Phase E2 entry point: construct trimmed faces for all HIGH-fit regions.

    For each region with a HIGH-confidence primitive fit, extracts the
    boundary edges from the mesh connectivity, orders them into closed
    loops, parameterizes each loop in surface-local UV space, and
    packages everything as a :class:`TrimmedFace`.

    Args:
        state: ReconstructionState populated by E0 + E1.
        full_mesh: trimesh.Trimesh (the snapped mesh if available, else
            the original cleaned mesh).

    Returns:
        Dict mapping region_id -> TrimmedFace, also stored on
        ``state.trimmed_faces``.
    """
    # Use snapped vertices when available (E1 output), otherwise raw mesh
    if state.snap_result is not None and hasattr(state.snap_result, "snapped_vertices"):
        vertices = np.asarray(state.snap_result.snapped_vertices, dtype=np.float64)
    else:
        vertices = np.asarray(full_mesh.vertices, dtype=np.float64)

    faces = np.asarray(full_mesh.faces, dtype=np.int64)

    trimmed: Dict[int, TrimmedFace] = {}
    n_skipped = 0

    for region_id, region in state.regions.items():
        # Only process HIGH-confidence fits
        if region.fit is None:
            continue
        if region.fit.confidence_class != ConfidenceClass.HIGH:
            continue
        if region.excluded:
            continue

        surface_type = region.fit.type
        if surface_type == PrimitiveType.UNKNOWN:
            continue

        params = dict(region.fit.params)

        # --- Extract boundary edges ---
        try:
            boundary_edges = _extract_boundary_edges(region.full_face_indices, faces)
        except Exception as exc:
            logger.warning(
                "Region %d: boundary edge extraction failed: %s", region_id, exc
            )
            n_skipped += 1
            continue

        if not boundary_edges:
            logger.debug(
                "Region %d: no boundary edges found (fully interior?), skipping.",
                region_id,
            )
            n_skipped += 1
            continue

        # --- Order into closed loops ---
        raw_loops = _order_boundary_loops(boundary_edges)
        if not raw_loops:
            logger.warning(
                "Region %d: could not form any closed boundary loops.", region_id
            )
            n_skipped += 1
            continue

        # --- Build BoundaryLoop objects ---
        loop_objects: List[BoundaryLoop] = []
        for loop_verts in raw_loops:
            if len(loop_verts) < 3:
                logger.debug(
                    "Region %d: dropping degenerate loop with %d vertices.",
                    region_id, len(loop_verts),
                )
                continue

            indices = np.array(loop_verts, dtype=np.int64)
            pts_3d = vertices[indices]  # (N, 3)

            # UV parameterization
            uv = _compute_uv(pts_3d, surface_type, params)

            loop_objects.append(
                BoundaryLoop(
                    vertices_3d=pts_3d,
                    vertex_indices=indices,
                    uv=uv,
                    is_outer=False,  # determined below
                )
            )

        if not loop_objects:
            logger.warning(
                "Region %d: all boundary loops were degenerate.", region_id
            )
            n_skipped += 1
            continue

        # --- Identify outer vs inner loops ---
        # The outer loop is the one enclosing the largest area.
        loop_areas = [_loop_area_3d(lp.vertices_3d) for lp in loop_objects]
        outer_idx = int(np.argmax(loop_areas))
        loop_objects[outer_idx].is_outer = True

        outer_loop = loop_objects[outer_idx]
        inner_loops = [lp for i, lp in enumerate(loop_objects) if i != outer_idx]

        # --- Compute region area from mesh faces ---
        # Sum the areas of the mesh triangles in this region
        region_faces_arr = faces[region.full_face_indices]  # (K, 3)
        v0 = vertices[region_faces_arr[:, 0]]
        v1 = vertices[region_faces_arr[:, 1]]
        v2 = vertices[region_faces_arr[:, 2]]
        tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        total_area = float(np.sum(tri_areas))

        # --- Assemble TrimmedFace ---
        tf = TrimmedFace(
            region_id=region_id,
            surface_type=surface_type.value,
            surface_params=params,
            outer_loop=outer_loop,
            inner_loops=inner_loops,
            area=total_area,
            n_mesh_faces=len(region.full_face_indices),
        )
        trimmed[region_id] = tf

    # Store on state
    state.trimmed_faces = trimmed

    logger.info(
        "E2 trimmed faces: %d constructed, %d skipped. %d with holes.",
        len(trimmed),
        n_skipped,
        sum(1 for tf in trimmed.values() if tf.inner_loops),
    )

    return trimmed


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def get_trim_summary(state: ReconstructionState) -> dict:
    """Return summary dict for the frontend.

    Provides counts and statistics about the trimmed faces currently
    stored on the reconstruction state.

    Args:
        state: ReconstructionState with trimmed_faces populated.

    Returns:
        Dict with keys: n_trimmed_faces, n_with_holes,
        total_boundary_vertices, by_surface_type, mean_boundary_vertices.
    """
    tf_map = state.trimmed_faces
    if not tf_map:
        return {
            "n_trimmed_faces": 0,
            "n_with_holes": 0,
            "total_boundary_vertices": 0,
            "by_surface_type": {},
            "mean_boundary_vertices": 0.0,
        }

    by_type: Dict[str, int] = {}
    total_bv = 0
    n_holes = 0

    for tf in tf_map.values():
        by_type[tf.surface_type] = by_type.get(tf.surface_type, 0) + 1
        bv = tf.outer_loop.n_vertices + sum(lp.n_vertices for lp in tf.inner_loops)
        total_bv += bv
        if tf.inner_loops:
            n_holes += 1

    n_total = len(tf_map)
    return {
        "n_trimmed_faces": n_total,
        "n_with_holes": n_holes,
        "total_boundary_vertices": total_bv,
        "by_surface_type": by_type,
        "mean_boundary_vertices": total_bv / n_total if n_total else 0.0,
    }

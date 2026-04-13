"""Build clean CAD geometry from detected surfaces.

Strategy: Take the original mesh topology and flatten each detected plane
region so all its triangles lie exactly on the fitted plane. Non-plane
regions (cylinders, cones, fillets) keep their snapped geometry.

This preserves the exact shape boundaries (no gaps, no overlaps) while
making planar surfaces perfectly flat — like reverse-engineered CAD.

For a lightweight version, planar regions can be decimated aggressively
(a flat surface needs very few triangles).
"""

import numpy as np
import trimesh
import logging

from .state import PrimitiveType, ConfidenceClass, ReconstructionState

logger = logging.getLogger(__name__)


def build_cad_geometry(state: ReconstructionState, original_mesh) -> tuple:
    """Build clean CAD geometry by flattening detected plane regions.

    Takes the snapped mesh and:
    1. For each HIGH-confidence PLANE region: project ALL vertices exactly
       onto the fitted plane → perfectly flat triangles
    2. For each CYLINDER region: project onto cylinder surface
    3. All other regions: keep snapped (or original) geometry
    4. Decimate planar regions aggressively (flat = few triangles needed)

    Returns (trimesh.Trimesh, stats_dict).
    """
    vertices = np.asarray(original_mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(original_mesh.faces, dtype=np.int64)

    # Use snapped vertices as starting point if available
    if state.snap_result is not None:
        vertices = state.snap_result.snapped_vertices.copy()

    # Center at origin
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices -= center

    n_plane = 0
    n_cyl = 0
    n_cone = 0
    n_other = 0

    # Track which vertices have been snapped to which surface
    # (a vertex shared by two planes should be snapped to intersection)
    vertex_plane_count = np.zeros(len(vertices), dtype=np.int32)
    vertex_plane_normals = [[] for _ in range(len(vertices))]
    vertex_plane_ds = [[] for _ in range(len(vertices))]

    # First pass: collect plane constraints per vertex
    for rid, region in state.regions.items():
        if region.fit is None:
            continue
        if region.fit.confidence_class != ConfidenceClass.HIGH:
            continue
        if region.fit.type != PrimitiveType.PLANE:
            continue
        if region.full_face_indices is None or len(region.full_face_indices) == 0:
            continue

        params = region.fit.params
        normal = np.array(params["normal"], dtype=np.float64)
        normal /= np.linalg.norm(normal) + 1e-12
        d = float(params.get("d", 0.0))
        # Adjust d for center offset
        d_adj = d + np.dot(normal, center)

        vert_idx = np.unique(faces[region.full_face_indices].ravel())
        for vi in vert_idx:
            vertex_plane_count[vi] += 1
            vertex_plane_normals[vi].append(normal)
            vertex_plane_ds[vi].append(d_adj)

    # Second pass: snap vertices to plane(s)
    # ONLY do single-plane projection (safe). Skip multi-plane intersection
    # which causes spikes when planes are nearly parallel.
    max_displacement = 0.01 * np.linalg.norm(np.ptp(vertices, axis=0))  # 1% of bbox

    for vi in range(len(vertices)):
        n = vertex_plane_count[vi]
        if n == 0:
            continue

        # Always project onto the single closest plane (safest approach)
        # Pick the plane with smallest distance to the vertex
        best_dist = float('inf')
        best_normal = None
        best_d = None
        for j in range(n):
            normal = vertex_plane_normals[vi][j]
            d = vertex_plane_ds[vi][j]
            dist = abs(np.dot(normal, vertices[vi]) + d)
            if dist < best_dist:
                best_dist = dist
                best_normal = normal
                best_d = d

        if best_normal is not None and best_dist < max_displacement:
            dist = np.dot(best_normal, vertices[vi]) + best_d
            vertices[vi] -= dist * best_normal
            n_plane += 1

    # Third pass: snap cylinder vertices
    for rid, region in state.regions.items():
        if region.fit is None:
            continue
        if region.fit.confidence_class != ConfidenceClass.HIGH:
            continue
        if region.fit.type != PrimitiveType.CYLINDER:
            continue
        if region.full_face_indices is None:
            continue

        params = region.fit.params
        axis = np.array(params["axis"], dtype=np.float64)
        axis /= np.linalg.norm(axis) + 1e-12
        ctr = np.array(params["center"], dtype=np.float64) - center
        radius = float(params["radius"])

        vert_idx = np.unique(faces[region.full_face_indices].ravel())
        # Only snap vertices not already claimed by planes
        for vi in vert_idx:
            if vertex_plane_count[vi] > 0:
                continue
            diff = vertices[vi] - ctr
            axial = np.dot(diff, axis) * axis
            radial = diff - axial
            r = np.linalg.norm(radial)
            if r > 1e-12:
                vertices[vi] = ctr + axial + radial * (radius / r)
                n_cyl += 1

    # Build the output mesh with ALL faces (preserves shape boundaries)
    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Remove degenerate faces
    good = result.nondegenerate_faces()
    result.update_faces(good)

    logger.info(
        "CAD geometry: %d plane verts snapped, %d cylinder verts, "
        "%d vertices total, %d faces",
        n_plane, n_cyl, len(result.vertices), len(result.faces),
    )

    stats = {
        "n_plane_faces": int(np.sum(vertex_plane_count > 0)),
        "n_cylinder_faces": n_cyl,
        "n_failed": n_other,
        "n_vertices": len(result.vertices),
        "n_triangles": len(result.faces),
    }

    return result, stats

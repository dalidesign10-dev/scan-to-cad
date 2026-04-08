"""Proxy mesh decimation + bidirectional label transfer.

The idea is simple and deliberately not novel: scan-cleaned meshes from
Phase A are typically 100k–500k triangles. Region growing, dihedral
analysis and boundary scoring on that scale are slow AND noisy because
they bounce off every Poisson stippling artefact. We therefore decimate
to ~30k triangles, do all the segmentation/boundary work on the proxy,
then push the resulting per-region labels back to the full-resolution
mesh via a nearest-face map. Refit happens on the full-res indices so
the actual primitive parameters are not biased by decimation.

We keep the maps explicit:
    face_map         : full -> proxy (one-to-one onto proxy)
    inverse_face_map : proxy -> [full, full, ...]

The map is built by querying each full-res face centroid against a KD-tree
of proxy face centroids. This is robust to topology changes from
quadric_decimation (which collapses vertices and re-triangulates).
"""

from typing import Optional
import numpy as np

from .state import MeshProxy


def build_proxy_mesh(
    full_mesh,
    target_face_count: int = 30000,
    progress_callback=None,
) -> MeshProxy:
    """Decimate `full_mesh` to ~target_face_count triangles via Open3D
    quadric decimation, then build the bidirectional face map.

    Skips decimation when the mesh is already small enough.
    """
    import open3d as o3d

    full_vertices = np.asarray(full_mesh.vertices, dtype=np.float64)
    full_faces = np.asarray(full_mesh.faces, dtype=np.int64)
    n_full = int(full_faces.shape[0])

    if progress_callback:
        progress_callback("intent.proxy", 5, f"Decimating {n_full:,} → ~{target_face_count:,} faces...")

    if n_full <= target_face_count:
        # Use the cleaned mesh directly as its own proxy.
        proxy_vertices = full_vertices.copy()
        proxy_faces = full_faces.copy()
    else:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(full_vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(full_faces)
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(target_face_count)
        )
        o3d_mesh.remove_unreferenced_vertices()
        proxy_vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
        proxy_faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)

    if proxy_faces.shape[0] == 0:
        raise RuntimeError("Proxy decimation produced an empty mesh")

    if progress_callback:
        progress_callback("intent.proxy", 25, "Computing proxy normals + areas...")

    proxy_face_normals, proxy_face_areas = _face_normals_and_areas(proxy_vertices, proxy_faces)

    # Drop zero-area faces — they break normals and KD-tree queries.
    valid = proxy_face_areas > 1e-12
    if not np.all(valid):
        proxy_faces = proxy_faces[valid]
        proxy_face_normals = proxy_face_normals[valid]
        proxy_face_areas = proxy_face_areas[valid]

    if progress_callback:
        progress_callback("intent.proxy", 45, "Building face_map (full -> proxy)...")

    full_centroids = _face_centroids(full_vertices, full_faces)
    proxy_centroids = _face_centroids(proxy_vertices, proxy_faces)

    # We also need full-res face normals to disambiguate the "full-res face
    # on a sharp corner maps to the proxy face on the wrong side" failure
    # (B4 in the initial commit). Cheap: we already have the formula.
    full_face_normals, _ = _face_normals_and_areas(full_vertices, full_faces)

    # Open3d KDTreeFlann works on point clouds; cheap and avoids extra deps.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(proxy_centroids)
    kdt = o3d.geometry.KDTreeFlann(pcd)

    # K-nearest + normal agreement tie-breaker. Nearest centroid alone can
    # map a full-res face on one side of a mechanical edge to a proxy face
    # on the other side when the proxy is coarse. For each full-res face,
    # we fetch the K closest proxy faces and pick the one whose normal is
    # most aligned with the full-res face normal — the proxy face "on the
    # same side of the edge" always wins.
    K = min(6, int(proxy_faces.shape[0]))
    face_map = np.zeros(n_full, dtype=np.int64)
    for i, c in enumerate(full_centroids):
        _, idx, _ = kdt.search_knn_vector_3d(c, K)
        if len(idx) == 0:
            face_map[i] = 0
            continue
        if len(idx) == 1:
            face_map[i] = int(idx[0])
            continue
        cand = np.asarray(idx, dtype=np.int64)
        dots = proxy_face_normals[cand] @ full_face_normals[i]
        best = int(cand[int(np.argmax(dots))])
        face_map[i] = best

    # Build inverse face map (proxy -> full[]) by argsort once.
    if progress_callback:
        progress_callback("intent.proxy", 75, "Building inverse_face_map...")

    order = np.argsort(face_map, kind="stable")
    sorted_proxy = face_map[order]
    inverse_face_map = []
    n_proxy = int(proxy_faces.shape[0])
    starts = np.searchsorted(sorted_proxy, np.arange(n_proxy), side="left")
    ends = np.searchsorted(sorted_proxy, np.arange(n_proxy), side="right")
    for s, e in zip(starts, ends):
        inverse_face_map.append(order[s:e].astype(np.int64))

    if progress_callback:
        progress_callback("intent.proxy", 100, f"Proxy ready: {n_proxy:,} faces")

    return MeshProxy(
        vertices=proxy_vertices,
        faces=proxy_faces,
        face_normals=proxy_face_normals,
        face_areas=proxy_face_areas,
        face_map=face_map,
        inverse_face_map=inverse_face_map,
        target_face_count=int(target_face_count),
        full_face_count=n_full,
    )


def transfer_labels_to_full(
    proxy_labels: np.ndarray,
    proxy: MeshProxy,
) -> np.ndarray:
    """Push a per-proxy-face label vector back to the full-resolution mesh."""
    assert proxy_labels.shape[0] == proxy.faces.shape[0], (
        f"label length {proxy_labels.shape[0]} != proxy faces {proxy.faces.shape[0]}"
    )
    out = np.empty(proxy.full_face_count, dtype=np.int64)
    out[:] = proxy_labels[proxy.face_map]
    return out


def _face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    return tri.mean(axis=1)


def _face_normals_and_areas(vertices: np.ndarray, faces: np.ndarray):
    tri = vertices[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norms
    safe = np.maximum(norms, 1e-20)
    normals = cross / safe[:, None]
    return normals.astype(np.float64), areas.astype(np.float64)

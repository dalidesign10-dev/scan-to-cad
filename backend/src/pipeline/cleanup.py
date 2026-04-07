"""Phase A — Scan cleanup via screened Poisson reconstruction.

Turns a noisy, non-watertight scan into a clean, watertight, manifold mesh
equivalent in quality to a hand-cleaned reference. Output is stored as
session["preprocessed"] so every downstream stage (segmentation, primitive
fitting, plane intersection, etc.) consumes it transparently.

Pipeline:
    1. Sample a dense point cloud from the input mesh (area-weighted).
    2. Estimate + globally orient point normals (Poisson needs consistent
       normals or it explodes).
    3. Run Open3D screened Poisson reconstruction.
    4. Trim low-density vertices (artifacts in regions far from samples).
    5. Manifold repair: remove duplicates, degenerates, non-manifold edges.
    6. Trimesh repair pass: fill holes, fix normals/winding, keep largest
       connected component.
    7. Optional light Taubin smoothing.
    8. Persist as session["preprocessed"] and write a transfer file.
"""
import uuid
import numpy as np
import trimesh
import open3d as o3d

from mesh_io_pkg.serialization import mesh_to_transfer_file


def _trimesh_to_o3d_pcd(mesh: trimesh.Trimesh, sample_count: int) -> o3d.geometry.PointCloud:
    """Sample the mesh surface and return an Open3D point cloud."""
    pts, _ = trimesh.sample.sample_surface(mesh, sample_count)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pcd


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return the largest connected component (by volume, falling back to face count)."""
    pieces = mesh.split(only_watertight=False)
    if len(pieces) <= 1:
        return mesh
    def score(p):
        try:
            v = abs(p.volume)
            if v > 0:
                return v
        except Exception:
            pass
        return float(len(p.faces))
    return max(pieces, key=score)


def cleanup_mesh(params, progress_callback=None, session=None):
    """Run the Phase A cleanup pipeline on session["mesh"].

    Params (all optional):
        poisson_depth      int   default 10  — Poisson octree depth (7..12)
        sample_count       int   default 400_000
        density_cutoff     float default 0.02 — drop bottom-quantile-density verts
        taubin_iters       int   default 5  — set 0 to disable smoothing
        keep_largest_only  bool  default True
    """
    mesh = session.get("mesh")
    if mesh is None:
        raise ValueError("No mesh loaded")

    poisson_depth = int(params.get("poisson_depth", 10))
    sample_count = int(params.get("sample_count", 400_000))
    # Default 0: no density trim. Trimming punches holes that break watertightness;
    # users who want tighter outlines can raise this.
    density_cutoff = float(params.get("density_cutoff", 0.0))
    taubin_iters = int(params.get("taubin_iters", 5))
    keep_largest_only = bool(params.get("keep_largest_only", True))

    # 1) Sample dense point cloud --------------------------------------------
    if progress_callback:
        progress_callback("cleanup", 5, f"Sampling {sample_count:,} surface points...")
    pcd = _trimesh_to_o3d_pcd(mesh, sample_count)

    # 2) Normals -------------------------------------------------------------
    if progress_callback:
        progress_callback("cleanup", 20, "Estimating point normals...")
    # Heuristic radius based on bounding box diagonal / sqrt(sample_count)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    radius = max(diag * 0.005, 1e-3)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 4, max_nn=30),
    )
    if progress_callback:
        progress_callback("cleanup", 35, "Orienting normals consistently...")
    try:
        pcd.orient_normals_consistent_tangent_plane(k=30)
    except Exception:
        # Fallback: orient toward centroid (less robust but never crashes)
        pcd.orient_normals_towards_camera_location(camera_location=np.asarray(mesh.centroid))

    # 3) Screened Poisson reconstruction -------------------------------------
    if progress_callback:
        progress_callback("cleanup", 50, f"Poisson reconstruction (depth={poisson_depth})...")
    o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=1.1, linear_fit=False,
    )
    densities = np.asarray(densities)

    # 4) Trim low-density artifacts -----------------------------------------
    if density_cutoff > 0 and len(densities) > 0:
        if progress_callback:
            progress_callback("cleanup", 65, f"Trimming low-density verts (cutoff={density_cutoff})...")
        threshold = np.quantile(densities, density_cutoff)
        keep_mask = densities >= threshold
        o3d_mesh.remove_vertices_by_mask(np.logical_not(keep_mask))

    # 5) Light Open3D cleanup (skip remove_non_manifold_edges — it punches
    #    holes which then prevent watertightness)
    if progress_callback:
        progress_callback("cleanup", 75, "Open3D dedup + degenerate removal...")
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.compute_vertex_normals()

    # 6) Convert to trimesh, keep largest, then aggressive hole fill --------
    verts = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    if len(faces) == 0:
        raise RuntimeError("Poisson reconstruction produced an empty mesh — try lower depth")
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    if keep_largest_only:
        if progress_callback:
            progress_callback("cleanup", 82, "Keeping largest component...")
        tri = _largest_component(tri)

    if progress_callback:
        progress_callback("cleanup", 86, "Trimesh hole fill + winding...")
    try:
        trimesh.repair.fix_normals(tri)
    except Exception:
        pass
    try:
        trimesh.repair.fix_winding(tri)
    except Exception:
        pass
    # Multiple hole-fill passes — Poisson edges sometimes leave staircased holes
    for _ in range(3):
        try:
            n_before = len(tri.faces)
            trimesh.repair.fill_holes(tri)
            if len(tri.faces) == n_before:
                break
        except Exception:
            break

    # 7) Optional Taubin smoothing ------------------------------------------
    if taubin_iters > 0:
        if progress_callback:
            progress_callback("cleanup", 92, f"Taubin smoothing ({taubin_iters} iters)...")
        try:
            trimesh.smoothing.filter_taubin(tri, lamb=0.5, nu=-0.53, iterations=taubin_iters)
        except Exception:
            pass

    # 8) Persist + write transfer file --------------------------------------
    if progress_callback:
        progress_callback("cleanup", 96, "Writing transfer file...")
    mesh_id = (session.get("mesh_id") or str(uuid.uuid4())[:8]) + "_clean"
    transfer_path = mesh_to_transfer_file(tri, session["temp_dir"], mesh_id)

    # Make the cleaned mesh the canonical input for downstream stages
    session["preprocessed"] = tri
    # Reset stale derived state so the user notices they need to re-run downstream
    session["labels"] = None
    session["patches"] = None
    session["primitives"] = None
    session.pop("labels_source", None)
    session.pop("patches_source", None)
    session.pop("primitives_source", None)

    if progress_callback:
        progress_callback("cleanup", 100, "Cleanup complete")

    try:
        volume = float(tri.volume)
    except Exception:
        volume = None

    return {
        "mesh_id": mesh_id,
        "transfer_path": transfer_path,
        "vertices": int(len(tri.vertices)),
        "faces": int(len(tri.faces)),
        "is_watertight": bool(tri.is_watertight),
        "volume": volume,
        "poisson_depth": poisson_depth,
        "sample_count": sample_count,
    }

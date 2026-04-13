"""Phase E3+E4 -- B-Rep construction + STEP export via pythonocc subprocess."""
import os
import sys
import json
import subprocess
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Path to the occ conda env Python
OCC_PYTHON = os.path.join(
    os.path.expanduser("~"), "miniconda3", "envs", "occ", "python.exe"
)
OCC_RUNNER = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "external", "occ_runner", "occ_runner.py"
    )
)

_SUBPROCESS_TIMEOUT = 300  # seconds (large models take time)


def _simplify_loop(vertices_3d, max_verts=80):
    """Simplify a boundary loop to at most max_verts using uniform subsampling."""
    pts = np.asarray(vertices_3d)
    n = len(pts)
    if n <= max_verts:
        return pts.tolist()
    # Uniform subsample keeping first point
    indices = np.linspace(0, n - 1, max_verts, dtype=int)
    return pts[indices].tolist()


def _jsonable_params(params):
    """Convert numpy arrays in params dict to plain Python lists."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _call_occ_runner(request: dict) -> dict:
    """Send JSON request to occ_runner subprocess, return result dict."""
    if not os.path.isfile(OCC_PYTHON):
        raise EnvironmentError(
            f"OCC conda environment Python not found at {OCC_PYTHON}. "
            "Please create it with: conda create -n occ python=3.10 pythonocc-core=7.8.1 -c conda-forge"
        )
    if not os.path.isfile(OCC_RUNNER):
        raise FileNotFoundError(f"occ_runner.py not found at {OCC_RUNNER}")

    payload = json.dumps(request).encode("utf-8")
    logger.debug("occ_runner request: op=%s, payload_size=%d bytes", request.get("op"), len(payload))

    try:
        proc = subprocess.run(
            [OCC_PYTHON, OCC_RUNNER],
            input=payload,
            capture_output=True,
            text=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"occ_runner subprocess timed out after {_SUBPROCESS_TIMEOUT}s "
            f"(op={request.get('op')})"
        )

    stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    stdout_text = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""

    if stderr_text:
        logger.warning("occ_runner stderr:\n%s", stderr_text[:500])

    if not stdout_text.strip():
        raise RuntimeError(
            f"occ_runner returned no output (exit code {proc.returncode}). stderr:\n{stderr_text[:500]}"
        )

    resp = json.loads(stdout_text)
    if not resp.get("ok"):
        raise RuntimeError(f"occ_runner error: {resp.get('error', 'unknown')}")

    return resp["result"]


def build_step_from_snapped_mesh(state, full_mesh, output_dir: str) -> dict:
    """Build a polyhedral STEP from the E1-snapped mesh.

    Uses occ_runner's 'build' op to create a sewed polyhedral solid from
    the snapped mesh triangles, then exports as STEP. This produces a
    good-looking solid that preserves the scan shape with clean geometry.
    """
    import trimesh

    if state is None or state.snap_result is None:
        raise ValueError("No snap result — run E1 first")

    os.makedirs(output_dir, exist_ok=True)

    # Decimate the snapped mesh to ~50K faces for OCC performance
    # (sewing 1M+ triangles takes too long)
    snapped_mesh = trimesh.Trimesh(
        vertices=state.snap_result.snapped_vertices,
        faces=state.snap_result.faces,
        process=False,
    )
    target_faces = min(15000, len(snapped_mesh.faces))
    if len(snapped_mesh.faces) > target_faces:
        logger.info("Decimating %d -> %d faces for STEP export",
                     len(snapped_mesh.faces), target_faces)
        try:
            import open3d as o3d
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(snapped_mesh.vertices))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(snapped_mesh.faces))
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
            snapped_mesh = trimesh.Trimesh(
                vertices=np.asarray(o3d_mesh.vertices),
                faces=np.asarray(o3d_mesh.triangles),
                process=False,
            )
        except Exception as e:
            logger.warning("Decimation failed (%s), using original mesh", e)
        logger.info("Decimated to %d faces", len(snapped_mesh.faces))

    mesh_path = os.path.join(output_dir, "snapped_for_step.stl")
    snapped_mesh.export(mesh_path)

    out_brep = os.path.join(output_dir, "snapped.brep")

    logger.info("Building polyhedral B-Rep from snapped mesh (%d verts, %d faces)",
                len(snapped_mesh.vertices), len(snapped_mesh.faces))

    # Step 1: Build polyhedral B-Rep from mesh
    build_result = _call_occ_runner({
        "op": "build",
        "mesh_path": mesh_path,
        "out_brep": out_brep,
    })

    # Step 2: Export as STEP
    out_step = os.path.join(output_dir, "output.step")
    export_result = _call_occ_runner({
        "op": "export",
        "shape_path": out_brep,
        "output_path": os.path.splitext(out_step)[0],
        "format": "step",
    })

    return {
        "n_faces_built": build_result.get("n_faces", 0),
        "n_faces_failed": 0,
        "brep_path": out_brep,
        "step_path": export_result.get("output_path", out_step),
        "step_size": export_result.get("size", 0),
        "mode": "polyhedral_snapped",
    }


def build_step_from_trimmed_faces(state, output_dir: str) -> dict:
    """Build OCC B-Rep from E2 trimmed faces and export as STEP.

    Args:
        state: ReconstructionState with trimmed_faces populated
        output_dir: directory for output files

    Returns:
        dict with step_path, brep_path, n_faces_built, n_faces_failed, step_size
    """
    if not state.trimmed_faces:
        raise ValueError("No trimmed faces -- run E2 first")

    os.makedirs(output_dir, exist_ok=True)

    from scipy.spatial import ConvexHull

    # For each plane region, collect ALL vertices, project onto the fitted
    # plane, compute 2D convex hull → clean polygon boundary.
    # Skip cylinders/cones for now — they need proper trimming.
    full_mesh_obj = None
    for mesh_key in ("preprocessed", "mesh"):
        # We need access to the full mesh to get vertex positions
        pass

    faces_json = []
    n_skipped = 0

    for tf in state.trimmed_faces.values():
        params = _jsonable_params(tf.surface_params)

        if tf.surface_type != "plane":
            n_skipped += 1
            continue

        try:
            normal = np.array(params["normal"], dtype=float)
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            d_val = float(params.get("d", 0.0))

            # Get ALL vertices of this region (not just boundary)
            region = state.regions.get(tf.region_id)
            if region is None or region.full_face_indices is None:
                continue

            # Use snapped vertices if available
            if state.snap_result is not None:
                all_verts = state.snap_result.snapped_vertices
            else:
                continue

            all_faces = state.snap_result.faces
            # Get unique vertex indices for this region's faces
            region_face_idx = region.full_face_indices
            region_vert_idx = np.unique(all_faces[region_face_idx].ravel())
            pts_3d = all_verts[region_vert_idx]

            if len(pts_3d) < 3:
                continue

            # Project onto the fitted plane
            dist = pts_3d @ normal + d_val
            projected = pts_3d - np.outer(dist, normal)

            # Build 2D local frame on the plane
            if abs(normal[0]) < 0.9:
                u_axis = np.cross(normal, [1, 0, 0])
            else:
                u_axis = np.cross(normal, [0, 1, 0])
            u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-12)
            v_axis = np.cross(normal, u_axis)

            centroid = projected.mean(axis=0)
            rel = projected - centroid
            pts_2d = np.column_stack([rel @ u_axis, rel @ v_axis])

            # Convex hull → clean polygon
            if len(pts_2d) < 3:
                continue
            hull = ConvexHull(pts_2d)
            hull_idx = hull.vertices  # indices into pts_2d

            # Convert hull back to 3D (on the plane)
            hull_3d = centroid + np.outer(pts_2d[hull_idx, 0], u_axis) + np.outer(pts_2d[hull_idx, 1], v_axis)

            # Simplify hull with Douglas-Peucker to get clean corners
            hull_3d_list = _simplify_loop(hull_3d, max_verts=60)

            if len(hull_3d_list) < 3:
                continue

            faces_json.append({
                "surface_type": "plane",
                "surface_params": params,
                "outer_loop": hull_3d_list,
                "inner_loops": [],
            })
        except Exception as e:
            logger.warning("Face %d failed: %s", tf.region_id, e)
            n_skipped += 1
            continue

    if not faces_json:
        raise ValueError("No exportable plane faces")

    logger.info("Built %d plane face boundaries, skipped %d non-plane", len(faces_json), n_skipped)

    out_brep = os.path.join(output_dir, "trimmed.brep")
    out_step = os.path.join(output_dir, "output.step")

    logger.info(
        "build_trimmed: %d faces -> %s", len(faces_json), out_step
    )

    result = _call_occ_runner({
        "op": "build_trimmed",
        "faces": faces_json,
        "out_brep": out_brep,
        "out_step": out_step,
    })

    logger.info(
        "build_trimmed complete: %d built, %d failed, STEP=%d bytes",
        result.get("n_faces_built", 0),
        result.get("n_faces_failed", 0),
        result.get("step_size", 0),
    )

    return result

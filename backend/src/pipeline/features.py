"""Phase C — Construction features (chamfer / fillet) on a polyhedral B-Rep.

OCC operations (`pythonocc-core`) live in a dedicated conda env (`occ`).
This module is a thin subprocess wrapper: it serializes the current mesh
to disk, calls the runner, and reads back the resulting tessellation.

Why subprocess: pythonocc-core is conda-only on Windows and would conflict
with the main backend's pip-installed dependencies. Same isolation pattern
we use for Point2Cyl.
"""
import os
import json
import subprocess
import tempfile
import numpy as np
import open3d as o3d
import trimesh


def _decimate_with_open3d(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    if len(o3d_mesh.triangles) > target_faces:
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )


OCC_PY = r"C:\Users\Dali Design\miniconda3\envs\occ\python.exe"
OCC_RUNNER = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "occ_runner", "occ_runner.py")
)


def _ensure_runner_available():
    if not os.path.exists(OCC_PY):
        raise RuntimeError(f"OCC python not found: {OCC_PY}")
    if not os.path.exists(OCC_RUNNER):
        raise RuntimeError(f"OCC runner not found: {OCC_RUNNER}")


def _run_runner(req: dict, timeout: int = 600):
    _ensure_runner_available()
    proc = subprocess.run(
        [OCC_PY, OCC_RUNNER],
        input=json.dumps(req),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if not proc.stdout:
        raise RuntimeError(
            f"OCC runner produced no stdout (rc={proc.returncode}). stderr: {proc.stderr[:1500]}"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"OCC runner returned non-JSON (rc={proc.returncode}): "
            f"stdout={proc.stdout[:500]} stderr={proc.stderr[:500]}"
        )
    if not result.get("ok"):
        raise RuntimeError(f"OCC runner error: {result.get('error', 'unknown')}")
    return result["result"]


def build_polyhedral_brep(params, progress_callback=None, session=None):
    """Decimate the cleaned mesh and lift it to a polyhedral OCC solid."""
    mesh = session.get("preprocessed") or session.get("mesh")
    if mesh is None:
        raise ValueError("No mesh loaded")

    target_faces = int(params.get("target_faces", 1500))

    if progress_callback:
        progress_callback("brep", 5, f"Decimating to {target_faces} faces...")

    decimated = _decimate_with_open3d(mesh, target_faces)

    if progress_callback:
        progress_callback("brep", 15, f"Exporting decimated mesh ({len(decimated.faces)} faces)...")

    tmp_dir = session["temp_dir"]
    mesh_stl = os.path.join(tmp_dir, "occ_input.stl")
    decimated.export(mesh_stl)

    out_brep = os.path.join(tmp_dir, "occ_shape.brep")

    if progress_callback:
        progress_callback("brep", 25, f"Running OCC build (target={target_faces})...")

    result = _run_runner({
        "op": "build",
        "mesh_path": mesh_stl,
        "target_faces": target_faces,
        "out_brep": out_brep,
    })

    session["brep_shape_path"] = result["brep_path"]
    if progress_callback:
        progress_callback("brep", 100, f"{result['n_faces']} faces in B-Rep")
    return result


def _apply_op(params, op_kind, progress_callback=None, session=None):
    shape_path = session.get("brep_shape_path")
    if not shape_path or not os.path.exists(shape_path):
        raise ValueError("Build the polyhedral B-Rep first")

    radius_or_dist_key = "radius" if op_kind == "fillet" else "distance"
    value = float(params.get(radius_or_dist_key, 0))
    if value <= 0:
        raise ValueError(f"{radius_or_dist_key} must be positive")
    min_dihedral = float(params.get("min_dihedral_deg", 20.0))

    out_brep = os.path.join(session["temp_dir"], f"occ_shape_{op_kind}.brep")

    if progress_callback:
        progress_callback("feat", 20, f"Running OCC {op_kind} subprocess...")

    req = {
        "op": op_kind,
        "shape_path": shape_path,
        "out_brep": out_brep,
        "min_dihedral_deg": min_dihedral,
        radius_or_dist_key: value,
    }
    result = _run_runner(req)
    session["brep_shape_path"] = result["brep_path"]
    if progress_callback:
        progress_callback("feat", 100, f"{op_kind} done — {result['n_edges_modified']} edges modified")
    return result


def fillet_sharp_edges(params, progress_callback=None, session=None):
    return _apply_op(params, "fillet", progress_callback, session)


def chamfer_sharp_edges(params, progress_callback=None, session=None):
    return _apply_op(params, "chamfer", progress_callback, session)


def export_brep(params, progress_callback=None, session=None):
    """Export the current Phase C B-Rep through OCC as STL / STEP / BREP."""
    shape_path = session.get("brep_shape_path")
    if not shape_path or not os.path.exists(shape_path):
        raise ValueError("Build the polyhedral B-Rep first")
    fmt = (params.get("format") or "stl").lower()
    if fmt not in ("stl", "step", "brep"):
        raise ValueError(f"Unsupported format: {fmt}")

    out_path = params.get("output_path")
    if not out_path:
        out_path = os.path.join(session["temp_dir"], f"brep_export.{fmt}")

    if progress_callback:
        progress_callback("export", 20, f"Exporting B-Rep as {fmt.upper()}...")

    result = _run_runner({
        "op": "export",
        "shape_path": shape_path,
        "output_path": out_path,
        "format": fmt,
    })

    if progress_callback:
        progress_callback("export", 100, f"Wrote {result['output_path']} ({result['size']} bytes)")
    return result

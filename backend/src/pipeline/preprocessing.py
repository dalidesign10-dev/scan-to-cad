"""Mesh preprocessing: load, denoise, fill holes."""
import uuid
import numpy as np
import trimesh
import open3d as o3d
from mesh_io_pkg.mesh_io import load_mesh_file
from mesh_io_pkg.serialization import mesh_to_transfer_file


def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def _o3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    verts = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def load_mesh(params, progress_callback=None, session=None):
    path = params["path"]
    if progress_callback:
        progress_callback("load", 10, "Loading mesh file...")

    mesh = load_mesh_file(path)
    mesh_id = str(uuid.uuid4())[:8]

    if progress_callback:
        progress_callback("load", 50, "Preparing transfer file...")

    transfer_path = mesh_to_transfer_file(mesh, session["temp_dir"], mesh_id)

    session["mesh"] = mesh
    session["mesh_id"] = mesh_id
    session["preprocessed"] = None
    session["labels"] = None
    session["patches"] = None
    session["primitives"] = None

    if progress_callback:
        progress_callback("load", 100, "Mesh loaded")

    return {
        "mesh_id": mesh_id,
        "transfer_path": transfer_path,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bounds_min": mesh.bounds[0].tolist(),
        "bounds_max": mesh.bounds[1].tolist(),
        "is_watertight": bool(mesh.is_watertight),
    }


def preprocess_mesh(params, progress_callback=None, session=None):
    mesh = session.get("mesh")
    if mesh is None:
        raise ValueError("No mesh loaded")

    method = params.get("denoise", "taubin")
    iterations = params.get("iterations", 10)
    fill_holes = params.get("fill_holes", True)

    if progress_callback:
        progress_callback("preprocess", 10, "Converting mesh...")

    o3d_mesh = _trimesh_to_o3d(mesh)

    if progress_callback:
        progress_callback("preprocess", 30, f"Smoothing ({method})...")

    if method == "taubin":
        o3d_mesh = o3d_mesh.filter_smooth_taubin(
            number_of_iterations=iterations,
            lambda_filter=0.5,
            mu=-0.53
        )
    elif method == "laplacian":
        o3d_mesh = o3d_mesh.filter_smooth_laplacian(
            number_of_iterations=iterations
        )

    o3d_mesh.compute_vertex_normals()

    if progress_callback:
        progress_callback("preprocess", 70, "Converting back...")

    result_mesh = _o3d_to_trimesh(o3d_mesh)

    if fill_holes:
        if progress_callback:
            progress_callback("preprocess", 80, "Filling holes...")
        result_mesh.fill_holes()

    mesh_id = session["mesh_id"] + "_pp"
    transfer_path = mesh_to_transfer_file(result_mesh, session["temp_dir"], mesh_id)

    session["preprocessed"] = result_mesh

    if progress_callback:
        progress_callback("preprocess", 100, "Preprocessing complete")

    return {
        "mesh_id": mesh_id,
        "transfer_path": transfer_path,
        "vertices": len(result_mesh.vertices),
        "faces": len(result_mesh.faces),
        "is_watertight": bool(result_mesh.is_watertight),
    }

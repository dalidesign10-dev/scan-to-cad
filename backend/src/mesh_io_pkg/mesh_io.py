"""Mesh I/O utilities using trimesh."""
import trimesh
import numpy as np


def load_mesh_file(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No triangle meshes found in file")
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def get_mesh_info(params, progress_callback=None, session=None):
    mesh = session.get("mesh") or session.get("preprocessed")
    if mesh is None:
        raise ValueError("No mesh loaded")
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bounds_min": mesh.bounds[0].tolist(),
        "bounds_max": mesh.bounds[1].tolist(),
        "is_watertight": bool(mesh.is_watertight),
    }

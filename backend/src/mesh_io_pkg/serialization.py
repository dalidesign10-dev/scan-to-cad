"""Serialize mesh data for transfer to the Electron frontend."""
import os
import struct
import numpy as np
import trimesh


def mesh_to_transfer_file(mesh: trimesh.Trimesh, output_dir: str, mesh_id: str) -> str:
    """Save mesh as a binary file optimized for Three.js loading.

    Format: header (3 uint32: nVerts, nFaces, hasNormals) +
            vertices (float32 * nVerts * 3) +
            normals (float32 * nVerts * 3) +
            faces (uint32 * nFaces * 3)
    """
    path = os.path.join(output_dir, f"{mesh_id}.bin")
    verts = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)

    with open(path, 'wb') as f:
        f.write(struct.pack('III', len(verts), len(faces), 1))
        f.write(verts.tobytes())
        f.write(normals.tobytes())
        f.write(faces.tobytes())

    return path


def labels_to_transfer_file(labels: np.ndarray, output_dir: str, mesh_id: str) -> str:
    """Save per-face labels as int32 binary file."""
    path = os.path.join(output_dir, f"{mesh_id}_labels.bin")
    labels.astype(np.int32).tofile(path)
    return path

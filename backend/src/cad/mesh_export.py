"""Export mesh as STL/OBJ/PLY.

Uses the snapped mesh from Phase E1 if available, otherwise the
preprocessed/raw mesh from the session.
"""
import os
import numpy as np
import trimesh


def export_cad_preview(params, progress_callback=None, session=None):
    """Export the current mesh (snapped or original) as STL/OBJ/PLY.

    Uses the E1 snapped mesh if available for cleaner geometry.
    """
    fmt = (params.get("format") or "stl").lower()
    output_path = params.get("output_path")
    if not output_path:
        raise ValueError("output_path required")

    if progress_callback:
        progress_callback("export", 20, "Preparing mesh for export...")

    # Prefer snapped mesh from E1, fall back to preprocessed or raw
    state = session.get("recon_state")
    mesh = session.get("preprocessed") or session.get("mesh")
    if mesh is None:
        raise ValueError("No mesh available")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if state is not None and state.snap_result is not None:
        # Use snapped vertices from Phase E1
        vertices = state.snap_result.snapped_vertices.copy()

    if progress_callback:
        progress_callback("export", 50, f"Writing {fmt}...")

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    tm.merge_vertices(merge_tex=False, merge_norm=True)

    base, _ = os.path.splitext(output_path)
    final_path = f"{base}.{fmt}"
    tm.export(final_path)

    if progress_callback:
        progress_callback("export", 100, f"Wrote {final_path}")

    return {
        "output_path": final_path,
        "format": fmt,
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(faces)),
        "snapped": state is not None and state.snap_result is not None,
    }

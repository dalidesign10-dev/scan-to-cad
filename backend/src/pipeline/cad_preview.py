"""Build a clean CAD-style preview by snapping each patch's vertices onto its
fitted analytic primitive. Returns per-patch trimmed surface meshes.

This is the visual end of the scan-to-CAD pipeline: the user sees what the
reconstructed CAD model would look like, with smooth analytic surfaces instead
of the raw scanned mesh.
"""
import numpy as np


# Type colors (R, G, B in 0-1 range), matching the palette
TYPE_COLOR = {
    "plane":    [0.31, 0.80, 0.64],
    "cylinder": [0.91, 0.27, 0.38],
    "sphere":   [0.96, 0.65, 0.14],
    "cone":     [0.55, 0.45, 0.85],
    "bspline":  [0.36, 0.50, 0.85],
}


def _snap_to_plane(points, normal, d):
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    proj = points - (points @ n + d)[:, None] * n[None, :]
    return proj, np.tile(n, (len(points), 1))


def _snap_to_cylinder(points, axis, center, radius):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    c = np.asarray(center, dtype=float)
    rel = points - c
    along = rel @ a
    radial = rel - along[:, None] * a[None, :]
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True)
    rnorm = np.where(rnorm < 1e-9, 1e-9, rnorm)
    radial_dir = radial / rnorm
    snapped = c + along[:, None] * a[None, :] + radius * radial_dir
    return snapped, radial_dir


def _snap_to_bspline(points, prim):
    """Snap to the fitted polynomial surface in the patch's local PCA frame."""
    centroid = np.asarray(prim["centroid"], dtype=float)
    u_axis = np.asarray(prim["u_axis"], dtype=float)
    v_axis = np.asarray(prim["v_axis"], dtype=float)
    w_axis = np.asarray(prim["w_axis"], dtype=float)
    coeffs = np.asarray(prim["poly_coeffs"], dtype=float)
    centered = points - centroid
    u = centered @ u_axis
    v = centered @ v_axis
    A = np.column_stack([
        u**4, u**3*v, u**2*v**2, u*v**3, v**4,
        u**3, u**2*v, u*v**2, v**3,
        u**2, u*v, v**2,
        u, v, np.ones(len(points)),
    ])
    w = A @ coeffs
    snapped = centroid + u[:, None] * u_axis + v[:, None] * v_axis + w[:, None] * w_axis
    return snapped, None


def _snap_to_sphere(points, center, radius):
    c = np.asarray(center, dtype=float)
    rel = points - c
    rnorm = np.linalg.norm(rel, axis=1, keepdims=True)
    rnorm = np.where(rnorm < 1e-9, 1e-9, rnorm)
    direction = rel / rnorm
    snapped = c + radius * direction
    return snapped, direction


def build_cad_preview(params, progress_callback=None, session=None):
    """For each fitted primitive, snap its patch's vertices onto the analytic
    surface and emit a small mesh (verts + faces + color)."""
    mesh = session.get("preprocessed") or session.get("mesh")
    primitives = session.get("primitives")
    labels = session.get("labels")
    fillet_features = session.get("fillet_features") or []
    if mesh is None or primitives is None or labels is None:
        raise ValueError("Need mesh, segmentation, and primitives first")

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = np.asarray(labels)

    # Map fillet patch_id -> neighbor primitives (for defeaturing)
    prim_by_pid = {p["patch_id"]: p for p in primitives}
    fillet_neighbors = {}
    for f in fillet_features:
        pid = f.get("patch_id")
        adj = f.get("adjacent_patches") or []
        neigh_prims = [prim_by_pid[a] for a in adj if a in prim_by_pid]
        if pid is not None and len(neigh_prims) >= 1:
            fillet_neighbors[pid] = neigh_prims

    if progress_callback:
        progress_callback("cad_preview", 10, f"Building preview for {len(primitives)} patches...")

    surfaces = []
    n_total = len(primitives)
    for i, prim in enumerate(primitives):
        if progress_callback and i % max(1, n_total // 10) == 0:
            progress_callback("cad_preview", 10 + int(80 * i / n_total),
                              f"Patch {i}/{n_total}")
        pid = prim["patch_id"]
        ptype = prim["type"]
        face_mask = labels == pid
        if not np.any(face_mask):
            continue
        patch_faces = faces[face_mask]
        # Get unique vertices for this patch and remap face indices
        vids, inverse = np.unique(patch_faces.ravel(), return_inverse=True)
        local_faces = inverse.reshape(patch_faces.shape)
        local_verts = vertices[vids]

        # Snap vertices onto the analytic primitive
        normals = None
        try:
            if pid in fillet_neighbors:
                # DEFEATURE: collapse fillet onto its underlying neighbor primitives
                neighs = fillet_neighbors[pid]
                # signed dist from each fillet vertex to each neighbor
                dist_list = []
                for np_prim in neighs:
                    nt = np_prim["type"]
                    if nt == "plane":
                        n = np.asarray(np_prim["normal"], dtype=float)
                        n /= np.linalg.norm(n)
                        d = np.abs(local_verts @ n + np_prim["d"])
                    elif nt == "cylinder":
                        a = np.asarray(np_prim["axis"], dtype=float)
                        a /= np.linalg.norm(a)
                        c = np.asarray(np_prim["center"], dtype=float)
                        rel = local_verts - c
                        radial = rel - (rel @ a)[:, None] * a[None, :]
                        d = np.abs(np.linalg.norm(radial, axis=1) - float(np_prim["radius"]))
                    elif nt == "sphere":
                        c = np.asarray(np_prim["center"], dtype=float)
                        d = np.abs(np.linalg.norm(local_verts - c, axis=1) - float(np_prim["radius"]))
                    else:
                        d = np.full(len(local_verts), np.inf)
                    dist_list.append(d)
                dists = np.stack(dist_list, axis=1)
                closest = np.argmin(dists, axis=1)
                snapped = np.empty_like(local_verts)
                for k, np_prim in enumerate(neighs):
                    mask = closest == k
                    if not np.any(mask):
                        continue
                    nt = np_prim["type"]
                    if nt == "plane":
                        s, _ = _snap_to_plane(local_verts[mask], np_prim["normal"], np_prim["d"])
                    elif nt == "cylinder":
                        s, _ = _snap_to_cylinder(local_verts[mask], np_prim["axis"],
                                                 np_prim["center"], np_prim["radius"])
                    elif nt == "sphere":
                        s, _ = _snap_to_sphere(local_verts[mask], np_prim["center"], np_prim["radius"])
                    else:
                        s = local_verts[mask]
                    snapped[mask] = s
            elif ptype == "plane":
                snapped, normals = _snap_to_plane(local_verts, prim["normal"], prim["d"])
            elif ptype == "cylinder":
                snapped, normals = _snap_to_cylinder(local_verts, prim["axis"],
                                                     prim["center"], prim["radius"])
            elif ptype == "sphere":
                snapped, normals = _snap_to_sphere(local_verts,
                                                   prim["center"], prim["radius"])
            elif ptype == "bspline" and "poly_coeffs" in prim:
                snapped, normals = _snap_to_bspline(local_verts, prim)
            else:
                snapped = local_verts
        except Exception:
            snapped = local_verts

        color = TYPE_COLOR.get(ptype, [0.5, 0.5, 0.5])
        surfaces.append({
            "patch_id": int(pid),
            "type": ptype,
            "vertices": snapped.astype(np.float32).tolist(),
            "faces": local_faces.astype(np.int32).tolist(),
            "color": color,
        })

    session["cad_preview"] = surfaces

    if progress_callback:
        progress_callback("cad_preview", 100, f"Built {len(surfaces)} CAD surfaces")

    type_counts = {}
    for s in surfaces:
        type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1

    return {
        "surfaces": surfaces,
        "n_surfaces": len(surfaces),
        "type_counts": type_counts,
    }

"""Feature detection: generate infinite/extended surfaces from significant patches.

This is the bridge between discrete segmented patches and the B-Rep reconstruction step.
For each large, well-fit patch, we generate an "infinite" analytic surface that can
later be intersected with adjacent surfaces to recover sharp edges.
"""
import numpy as np


def _compute_mesh_bbox_size(mesh):
    bbox = np.ptp(mesh.vertices, axis=0)
    return float(np.linalg.norm(bbox))


def _build_infinite_plane(prim, extent):
    """Build an infinite plane surface extended across the model bbox."""
    normal = np.array(prim["normal"])
    d = prim["d"]
    centroid = np.array(prim.get("centroid", [0, 0, 0]))

    # Project centroid onto the plane
    plane_point = centroid - (np.dot(normal, centroid) + d) * normal

    # Build an orthonormal basis on the plane
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Four corners of an "infinite" plane (moderately extended — not overwhelming)
    half = extent * 0.25
    corners = [
        plane_point + half * u + half * v,
        plane_point - half * u + half * v,
        plane_point - half * u - half * v,
        plane_point + half * u - half * v,
    ]

    return {
        "type": "infinite_plane",
        "patch_id": prim["patch_id"],
        "face_count": prim.get("face_count", 0),
        "normal": normal.tolist(),
        "d": float(d),
        "point": plane_point.tolist(),
        "u_axis": u.tolist(),
        "v_axis": v.tolist(),
        "corners": [c.tolist() for c in corners],
        "extent": float(extent),
    }


def _build_infinite_cylinder(prim, extent):
    """Build an extended cylinder surface.

    The cylinder is rendered as a mesh of length = extent, aligned along its axis.
    """
    axis = np.array(prim["axis"])
    axis /= np.linalg.norm(axis)
    center = np.array(prim.get("centroid", prim.get("center", [0, 0, 0])))
    radius = float(prim["radius"])

    # Project center onto the cylinder axis (using the fit center)
    fit_center = np.array(prim.get("center", center))
    # Axis passes through fit_center; extend along axis
    length = extent * 0.5

    return {
        "type": "infinite_cylinder",
        "patch_id": prim["patch_id"],
        "face_count": prim.get("face_count", 0),
        "axis": axis.tolist(),
        "center": fit_center.tolist(),
        "radius": radius,
        "length": float(length),
    }


def _build_infinite_sphere(prim, extent):
    """Build a sphere surface (spheres are naturally bounded)."""
    center = np.array(prim.get("center", prim.get("centroid", [0, 0, 0])))
    radius = float(prim["radius"])
    return {
        "type": "infinite_sphere",
        "patch_id": prim["patch_id"],
        "face_count": prim.get("face_count", 0),
        "center": center.tolist(),
        "radius": radius,
    }


def detect_features(params, progress_callback=None, session=None):
    """Convert significant primitives into infinite/extended surfaces.

    A 'big segment' is one with face_count >= min_faces AND a good primitive fit
    (inlier_ratio >= min_inlier). Each becomes an analytic infinite surface.
    """
    mesh = session.get("preprocessed") or session.get("mesh")
    primitives = session.get("primitives")
    patches = session.get("patches")

    if mesh is None or primitives is None:
        raise ValueError("Must fit primitives before detecting features")

    min_faces = params.get("min_faces", 150)
    min_inlier = params.get("min_inlier", 0.4)

    if progress_callback:
        progress_callback("features", 20, "Computing bbox and filtering big segments...")

    extent = _compute_mesh_bbox_size(mesh)

    # Filter for significant primitives
    big_primitives = [
        p for p in primitives
        if p.get("face_count", 0) >= min_faces
        and p.get("inlier_ratio", 0) >= min_inlier
        and p.get("type") in ("plane", "cylinder", "sphere")
    ]

    if progress_callback:
        progress_callback("features", 50,
                          f"Building {len(big_primitives)} infinite surfaces...")

    infinite_surfaces = []
    for prim in big_primitives:
        ptype = prim["type"]
        try:
            if ptype == "plane":
                surf = _build_infinite_plane(prim, extent)
            elif ptype == "cylinder":
                surf = _build_infinite_cylinder(prim, extent)
            elif ptype == "sphere":
                surf = _build_infinite_sphere(prim, extent)
            else:
                continue
            infinite_surfaces.append(surf)
        except Exception as e:
            continue

    # Also keep any fillet bands for reference
    fillet_primitives = [p for p in primitives if p.get("is_fillet")]

    # Legacy features list: keep empty/simple entries for backward compat
    features = []
    for surf in infinite_surfaces:
        features.append({
            "type": surf["type"],
            "patch_id": surf["patch_id"],
            "face_count": surf["face_count"],
        })
    for fp in fillet_primitives:
        features.append({
            "type": "fillet",
            "patch_id": fp["patch_id"],
            "face_count": fp.get("face_count", 0),
            "estimated_radius": fp.get("radius"),
        })

    session["features"] = features
    session["infinite_surfaces"] = infinite_surfaces

    if progress_callback:
        progress_callback("features", 100, "Feature detection complete")

    type_counts = {}
    for s in infinite_surfaces:
        t = s["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "features": features,
        "infinite_surfaces": infinite_surfaces,
        "type_counts": type_counts,
        "n_big_surfaces": len(infinite_surfaces),
        "n_fillets": len(fillet_primitives),
    }

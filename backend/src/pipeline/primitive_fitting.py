"""Primitive and surface fitting for segmented patches.

Uses the segmentation classification as a hint and estimates parameters
for the suggested type, falling back to other types if the fit is poor.
"""
import numpy as np
from utils.geometry import fit_plane_lstsq, fit_sphere_lstsq
from utils.nurbs_fitting import fit_bspline_surface, classify_surface_type


def _fit_plane(points, normals):
    """Direct least-squares plane fit. Returns dict or None."""
    if len(points) < 3:
        return None
    try:
        normal, d = fit_plane_lstsq(points)
    except (np.linalg.LinAlgError, ValueError):
        return None

    # Make sure plane normal aligns with average vertex normal
    avg_n = normals.mean(axis=0)
    if np.dot(normal, avg_n) < 0:
        normal = -normal
        d = -d

    # Compute fit quality
    distances = np.abs(points @ normal + d)
    bbox_diag = np.linalg.norm(np.ptp(points, axis=0))
    threshold = max(bbox_diag * 0.02, 0.1)
    inliers = np.sum(distances < threshold)
    inlier_ratio = inliers / len(points)
    rmse = float(np.sqrt(np.mean(distances ** 2)))

    return {
        "type": "plane",
        "normal": normal.tolist(),
        "d": float(d),
        "inlier_ratio": float(inlier_ratio),
        "rmse": rmse,
    }


def _fit_cylinder(points, normals):
    """Cylinder fit using SVD of normals to find axis, then circle fit.

    Works well for patches already classified as cylindrical.
    """
    n = len(points)
    if n < 10:
        return None

    # Cylinder axis is the direction perpendicular to the normal-spread plane
    # i.e., the smallest-variance direction of the normals
    normals_centered = normals - normals.mean(axis=0)
    try:
        _, s, vh = np.linalg.svd(normals_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    axis = vh[2]
    axis /= np.linalg.norm(axis)

    # Project points to plane perpendicular to axis
    centroid = points.mean(axis=0)
    diff = points - centroid
    proj_along = (diff @ axis)[:, None] * axis
    radial = diff - proj_along

    # 2D coordinates in the plane perpendicular to axis
    u_ax = vh[0]
    v_ax = vh[1]
    x2d = radial @ u_ax
    y2d = radial @ v_ax

    # Algebraic circle fit: x^2 + y^2 + Dx + Ey + F = 0
    A = np.column_stack([x2d, y2d, np.ones(n)])
    b = -(x2d ** 2 + y2d ** 2)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        D, E, F = sol
        cx = -D / 2
        cy = -E / 2
        r_squared = cx ** 2 + cy ** 2 - F
        if r_squared <= 0:
            return None
        radius = float(np.sqrt(r_squared))
    except np.linalg.LinAlgError:
        return None

    if radius < 0.01 or radius > 1e5:
        return None

    # Reject if radius is much larger than the patch extent
    # (indicates an ill-conditioned fit on near-flat data)
    bbox_diag = np.linalg.norm(np.ptp(points, axis=0))
    if radius > bbox_diag * 1.5:
        return None

    center = centroid + cx * u_ax + cy * v_ax

    # Compute fit quality: distance from each point to the cylinder surface
    diff_c = points - center
    proj_c = (diff_c @ axis)[:, None] * axis
    radial_c = diff_c - proj_c
    actual_dists = np.linalg.norm(radial_c, axis=1)
    residuals = np.abs(actual_dists - radius)
    bbox_diag = np.linalg.norm(np.ptp(points, axis=0))
    threshold = max(bbox_diag * 0.02, 0.1)
    inliers = np.sum(residuals < threshold)
    inlier_ratio = inliers / n
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Cylinder height along axis
    proj_heights = diff @ axis
    height = float(proj_heights.max() - proj_heights.min())

    return {
        "type": "cylinder",
        "axis": axis.tolist(),
        "center": center.tolist(),
        "radius": radius,
        "height": height,
        "inlier_ratio": float(inlier_ratio),
        "rmse": rmse,
    }


def _fit_sphere(points):
    """Direct least-squares sphere fit."""
    if len(points) < 4:
        return None
    try:
        center, radius = fit_sphere_lstsq(points)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if radius <= 0 or radius > 1e6:
        return None

    # Reject sphere fits where radius is much larger than patch (ill-conditioned)
    bbox_diag = np.linalg.norm(np.ptp(points, axis=0))
    if radius > bbox_diag * 1.5:
        return None

    distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    threshold = max(bbox_diag * 0.02, 0.1)
    inliers = np.sum(distances < threshold)
    inlier_ratio = inliers / len(points)
    rmse = float(np.sqrt(np.mean(distances ** 2)))

    return {
        "type": "sphere",
        "center": center.tolist(),
        "radius": float(radius),
        "inlier_ratio": float(inlier_ratio),
        "rmse": rmse,
    }


def _compute_patch_bounds(points):
    centroid = points.mean(axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    size = float(np.linalg.norm(bbox_max - bbox_min))
    return centroid.tolist(), bbox_min.tolist(), bbox_max.tolist(), size


def fit_single_patch(points, normals, classification_hint=None):
    """Fit the best surface type to a patch.

    Uses classification_hint from segmentation as the primary guess,
    then tries other types if the fit is poor.
    """
    candidates = []

    # Build candidate fits based on the segmentation hint
    type_priority = []
    if classification_hint == "planar":
        type_priority = ["plane", "cylinder", "sphere", "bspline"]
    elif classification_hint == "cylindrical":
        # Never try plane for cylindrical patches — plane always has ~1.0 inlier_ratio on nearly-flat cylinders
        type_priority = ["cylinder", "sphere", "bspline"]
    elif classification_hint == "spherical":
        type_priority = ["sphere", "bspline", "cylinder"]
    elif classification_hint == "conical":
        type_priority = ["cylinder", "sphere", "bspline"]
    elif classification_hint == "curved":
        type_priority = ["cylinder", "sphere", "bspline"]
    elif classification_hint == "fillet":
        # Fillets are cylinders in concept (rolling ball surface) — fit as cylinder, fall back to bspline
        type_priority = ["cylinder", "bspline"]
    else:
        type_priority = ["bspline", "cylinder", "sphere", "plane"]

    # Try the suggested type first
    fit_funcs = {
        "plane": lambda: _fit_plane(points, normals),
        "cylinder": lambda: _fit_cylinder(points, normals),
        "sphere": lambda: _fit_sphere(points),
        "bspline": lambda: fit_bspline_surface(points, normals),
    }

    primary = type_priority[0]
    primary_fit = fit_funcs[primary]()

    # Acceptance criteria per type
    if primary == "plane" and primary_fit and primary_fit["inlier_ratio"] >= 0.8:
        return primary_fit
    if primary == "cylinder" and primary_fit and primary_fit["inlier_ratio"] >= 0.4:
        return primary_fit
    if primary == "sphere" and primary_fit and primary_fit["inlier_ratio"] >= 0.4:
        return primary_fit
    if primary == "bspline" and primary_fit:
        bbox_diag = np.linalg.norm(np.ptp(points, axis=0))
        threshold = max(bbox_diag * 0.02, 0.1)
        primary_fit["inlier_ratio"] = float(max(0, 1.0 - primary_fit.get("rmse", 999) / threshold))
        primary_fit["surface_class"] = classify_surface_type(primary_fit, points, normals)

    if primary_fit:
        candidates.append(primary_fit)

    # Try alternates if primary failed
    if not primary_fit or primary_fit.get("inlier_ratio", 0) < 0.6:
        for alt_type in type_priority[1:]:
            alt_fit = fit_funcs[alt_type]()
            if alt_fit:
                if alt_type == "bspline":
                    bbox_diag = np.linalg.norm(np.ptp(points, axis=0))
                    threshold = max(bbox_diag * 0.02, 0.1)
                    alt_fit["inlier_ratio"] = float(max(0, 1.0 - alt_fit.get("rmse", 999) / threshold))
                    alt_fit["surface_class"] = classify_surface_type(alt_fit, points, normals)
                candidates.append(alt_fit)
                # Stop early if we get a great fit
                if alt_fit.get("inlier_ratio", 0) > 0.85:
                    break

    if not candidates:
        return {"type": "freeform", "inlier_ratio": 0}

    # Sort by inlier ratio with type preference (simpler primitives win ties)
    type_pref_score = {"plane": 0.05, "cylinder": 0.04, "sphere": 0.03, "bspline": 0.0, "freeform": -0.1}

    def score(c):
        return c.get("inlier_ratio", 0) + type_pref_score.get(c["type"], 0)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def fit_primitives(params, progress_callback=None, session=None):
    mesh = session.get("preprocessed") or session.get("mesh")
    labels = session.get("labels")
    patches_info = session.get("patches")

    if mesh is None or labels is None:
        raise ValueError("Must segment before fitting primitives")

    vertices = mesh.vertices
    faces = mesh.faces
    vertex_normals = mesh.vertex_normals

    primitives = []
    n_patches = len(patches_info)

    for i, patch in enumerate(patches_info):
        if progress_callback and i % max(1, n_patches // 20) == 0:
            pct = int(10 + 80 * i / n_patches)
            progress_callback("fit", pct, f"Fitting patch {i+1}/{n_patches}...")

        mask = labels == i
        face_indices = np.where(mask)[0]
        if len(face_indices) < 3:
            primitives.append({"patch_id": i, "type": "freeform", "inlier_ratio": 0})
            continue

        vert_indices = np.unique(faces[face_indices].flatten())
        pts = vertices[vert_indices]
        norms = vertex_normals[vert_indices]

        # Use classification from segmentation as hint
        hint = patch.get("classification", "unknown")
        is_fillet = patch.get("is_fillet", False) or hint == "fillet"
        result = fit_single_patch(pts, norms, classification_hint=hint)
        result["patch_id"] = i
        result["face_count"] = int(patch["face_count"])
        result["seg_class"] = hint
        if is_fillet:
            # Mark as fillet so UI/feature detection can pick it up
            result["is_fillet"] = True
            # Also label the type as "fillet" in the counts even if fitted as cylinder
            result["feature_type"] = "fillet"

        centroid, bbox_min, bbox_max, size = _compute_patch_bounds(pts)
        result["centroid"] = centroid
        result["bbox_min"] = bbox_min
        result["bbox_max"] = bbox_max
        result["patch_size"] = size

        primitives.append(result)

    session["primitives"] = primitives

    if progress_callback:
        progress_callback("fit", 100, "Surface fitting complete")

    type_counts = {}
    for p in primitives:
        t = p["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "primitives": primitives,
        "type_counts": type_counts,
    }

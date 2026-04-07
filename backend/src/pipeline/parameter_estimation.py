"""Parameter estimation for detected features."""
import numpy as np
from utils.geometry import fit_sphere_lstsq


def estimate_fillet_radius(mesh, labels, feature):
    """Refine fillet radius estimate using sphere fitting on fillet vertices."""
    faces = mesh.faces
    mask = labels == feature["patch_id"]
    face_indices = np.where(mask)[0]
    vert_indices = np.unique(faces[face_indices].flatten())
    points = mesh.vertices[vert_indices]

    if len(points) < 4:
        return feature.get("estimated_radius")

    try:
        center, radius = fit_sphere_lstsq(points)
        return float(radius)
    except (np.linalg.LinAlgError, ValueError):
        return feature.get("estimated_radius")


def estimate_chamfer_width(mesh, labels, feature):
    """Estimate chamfer width from the patch geometry."""
    faces = mesh.faces
    mask = labels == feature["patch_id"]
    face_indices = np.where(mask)[0]
    vert_indices = np.unique(faces[face_indices].flatten())
    points = mesh.vertices[vert_indices]

    if len(points) < 3:
        return None

    # Use PCA to find the narrow direction
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    # Smallest singular value corresponds to width
    width = 2 * s[-1] / np.sqrt(len(points))
    return float(width)

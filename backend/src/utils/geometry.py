"""Shared geometry utilities."""
import numpy as np


def fit_plane_lstsq(points):
    """Fit a plane to points using least-squares. Returns (normal, d)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    d = -np.dot(normal, centroid)
    return normal, d


def fit_sphere_lstsq(points):
    """Fit a sphere to points. Returns (center, radius)."""
    A = np.column_stack([2 * points, np.ones(len(points))])
    b = np.sum(points ** 2, axis=1)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center = result[:3]
    radius = np.sqrt(result[3] + np.sum(center ** 2))
    return center, radius


def fit_cylinder_lstsq(points, normals):
    """Estimate cylinder axis and radius from points+normals.

    Uses PCA of normals to get axis direction, then fits radius.
    Returns (axis, center, radius).
    """
    # Axis is perpendicular to normals
    _, _, vh = np.linalg.svd(normals - normals.mean(axis=0), full_matrices=False)
    axis = vh[-1]
    axis /= np.linalg.norm(axis)

    # Project points onto plane perpendicular to axis
    centroid = points.mean(axis=0)
    projected = points - centroid
    projected -= np.outer(projected @ axis, axis)

    # Fit circle in 2D
    dists = np.linalg.norm(projected, axis=1)
    radius = np.median(dists)

    return axis, centroid, radius


def plane_plane_intersection(n1, d1, n2, d2):
    """Intersection line of two planes. Returns (point, direction)."""
    direction = np.cross(n1, n2)
    denom = np.linalg.norm(direction)
    if denom < 1e-10:
        return None, None
    direction /= denom

    # Find a point on the line
    A = np.array([n1, n2, direction])
    b = np.array([-d1, -d2, 0.0])
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None

    return point, direction

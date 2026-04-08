"""First-pass primitive fitting for the intent layer.

Strict scope for E0:
    - plane
    - cylinder
    - unknown

The fitter is honest:
    - it always reports residuals + inlier ratio + a 95th percentile residual
    - it grades the fit into a discrete ConfidenceClass
    - if the fit fails any meaningful gate it returns UNKNOWN

We do NOT touch cones, spheres, freeform, fillets, B-splines.

Scale awareness: every numeric threshold is expressed relative to the
region's own bbox diagonal, so the same code works on a 10mm boss and a
500mm bracket without retuning.
"""

from typing import Optional
import numpy as np

from .state import (
    PrimitiveFit,
    PrimitiveType,
    ConfidenceClass,
)


# Acceptance gates (relative to bbox diagonal of the region's points)
PLANE_HIGH_RMSE_REL = 0.005
PLANE_MED_RMSE_REL = 0.012
PLANE_HIGH_INLIER = 0.85
PLANE_MED_INLIER = 0.65

CYL_HIGH_RMSE_REL = 0.010
CYL_MED_RMSE_REL = 0.025
CYL_HIGH_INLIER = 0.75
CYL_MED_INLIER = 0.55

# Cylinder rejection gates (independent of confidence class)
CYL_MIN_RADIUS_REL = 0.02     # smaller radius than 2% of bbox is almost certainly noise-of-a-flat
CYL_MAX_RADIUS_REL = 8.0      # 8x bbox is the unfit-flat-as-cylinder failure mode


def fit_region(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    fit_source: str = "proxy",
) -> PrimitiveFit:
    """Try plane then cylinder; return whichever wins, else UNKNOWN.

    The decision is NOT just "best inlier ratio" — a plane that passes the
    HIGH gate always wins, even if the cylinder has marginally lower RMSE
    (mechanical bias: planar > cylindrical when ambiguous, because flat
    surfaces dominate molded brackets and a wrong cylinder is uglier).
    """
    if points.shape[0] < 8:
        return _unknown(np.array([]), fit_source, "too few points")

    bbox_diag = float(np.linalg.norm(np.ptp(points, axis=0)))
    if bbox_diag < 1e-9:
        return _unknown(np.array([]), fit_source, "degenerate bbox")

    plane = _fit_plane(points, normals, bbox_diag, fit_source)

    if plane.confidence_class == ConfidenceClass.HIGH:
        return plane

    cylinder = _fit_cylinder(points, normals, bbox_diag, fit_source)

    candidates = [c for c in (plane, cylinder) if c is not None]
    candidates = [c for c in candidates if c.confidence_class != ConfidenceClass.REJECTED]

    if not candidates:
        return _unknown(plane_residuals(points, plane), fit_source, "no candidate passed")

    # Prefer the higher confidence class. Within the same class, prefer the
    # one with the lower RMSE relative to bbox diagonal.
    class_order = {
        ConfidenceClass.HIGH: 3,
        ConfidenceClass.MEDIUM: 2,
        ConfidenceClass.LOW: 1,
        ConfidenceClass.REJECTED: 0,
    }
    candidates.sort(
        key=lambda c: (class_order[c.confidence_class], -c.rmse / bbox_diag),
        reverse=True,
    )
    best = candidates[0]
    if best.confidence_class == ConfidenceClass.LOW:
        # Don't trust LOW fits — keep the region honest as UNKNOWN.
        return _unknown(np.array([best.rmse]), fit_source, f"best fit only LOW ({best.type.value})")
    return best


def plane_residuals(points: np.ndarray, fit: PrimitiveFit) -> np.ndarray:
    if fit.type != PrimitiveType.PLANE:
        return np.array([])
    n = np.asarray(fit.params["normal"], dtype=np.float64)
    d = float(fit.params["d"])
    return np.abs(points @ n + d)


def _fit_plane(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    bbox_diag: float,
    fit_source: str,
) -> PrimitiveFit:
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return _rejected(PrimitiveType.PLANE, fit_source, "SVD failed")
    normal = vh[-1].astype(np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)

    if normals is not None and len(normals):
        avg_n = normals.mean(axis=0)
        if float(np.dot(normal, avg_n)) < 0:
            normal = -normal

    d = float(-np.dot(normal, centroid))
    residuals = np.abs(points @ normal + d)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    p95 = float(np.quantile(residuals, 0.95))

    threshold = max(bbox_diag * PLANE_HIGH_RMSE_REL, 1e-9)
    inliers = float(np.mean(residuals < threshold))

    cls = _grade_plane(rmse, inliers, bbox_diag)
    score = _plane_score(rmse, inliers, bbox_diag)
    return PrimitiveFit(
        type=PrimitiveType.PLANE,
        params={
            "normal": normal.tolist(),
            "d": d,
            "centroid": centroid.tolist(),
        },
        rmse=rmse,
        inlier_ratio=inliers,
        residual_p95=p95,
        confidence_class=cls,
        score=score,
        fit_source=fit_source,
    )


def _fit_cylinder(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    bbox_diag: float,
    fit_source: str,
) -> PrimitiveFit:
    if normals is None or normals.shape[0] < 8:
        return _rejected(PrimitiveType.CYLINDER, fit_source, "no normals")

    # Axis ≈ smallest singular vector of the normal cloud.
    nc = normals - normals.mean(axis=0)
    try:
        _, sv, vh = np.linalg.svd(nc, full_matrices=False)
    except np.linalg.LinAlgError:
        return _rejected(PrimitiveType.CYLINDER, fit_source, "SVD failed")

    if sv[0] < 1e-9:
        return _rejected(PrimitiveType.CYLINDER, fit_source, "degenerate normals")

    # If the smallest singular value is too LARGE relative to the largest, the
    # normals don't lie on a great circle and a cylinder fit is meaningless.
    sv_ratio = float(sv[2] / sv[0])
    if sv_ratio > 0.45:
        return _rejected(PrimitiveType.CYLINDER, fit_source, f"normals not coplanar (sv_ratio={sv_ratio:.2f})")

    axis = vh[2].astype(np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)

    centroid = points.mean(axis=0)
    diff = points - centroid
    perp = diff - np.outer(diff @ axis, axis)
    u_ax = vh[0].astype(np.float64)
    v_ax = vh[1].astype(np.float64)
    x2 = perp @ u_ax
    y2 = perp @ v_ax

    # Algebraic circle fit in the plane perpendicular to axis.
    A = np.column_stack([x2, y2, np.ones_like(x2)])
    b = -(x2 ** 2 + y2 ** 2)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return _rejected(PrimitiveType.CYLINDER, fit_source, "circle lstsq failed")

    D, E, F = sol
    cx = -D / 2.0
    cy = -E / 2.0
    r2 = cx ** 2 + cy ** 2 - F
    if r2 <= 0:
        return _rejected(PrimitiveType.CYLINDER, fit_source, "negative radius^2")
    radius = float(np.sqrt(r2))

    if radius < bbox_diag * CYL_MIN_RADIUS_REL:
        return _rejected(PrimitiveType.CYLINDER, fit_source, f"radius too small ({radius:.4f})")
    if radius > bbox_diag * CYL_MAX_RADIUS_REL:
        return _rejected(PrimitiveType.CYLINDER, fit_source, f"radius too large ({radius:.4f})")

    center = centroid + cx * u_ax + cy * v_ax

    diff_c = points - center
    proj_c = (diff_c @ axis)[:, None] * axis
    radial = np.linalg.norm(diff_c - proj_c, axis=1)
    residuals = np.abs(radial - radius)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    p95 = float(np.quantile(residuals, 0.95))

    threshold = max(bbox_diag * CYL_HIGH_RMSE_REL, 1e-9)
    inliers = float(np.mean(residuals < threshold))

    proj_heights = (diff_c @ axis)
    height = float(proj_heights.max() - proj_heights.min())

    cls = _grade_cylinder(rmse, inliers, bbox_diag)
    score = _cylinder_score(rmse, inliers, bbox_diag)
    return PrimitiveFit(
        type=PrimitiveType.CYLINDER,
        params={
            "axis": axis.tolist(),
            "center": center.tolist(),
            "radius": radius,
            "height": height,
        },
        rmse=rmse,
        inlier_ratio=inliers,
        residual_p95=p95,
        confidence_class=cls,
        score=score,
        fit_source=fit_source,
    )


def _unknown(residuals: np.ndarray, fit_source: str, note: str) -> PrimitiveFit:
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    p95 = float(np.quantile(residuals, 0.95)) if residuals.size else 0.0
    return PrimitiveFit(
        type=PrimitiveType.UNKNOWN,
        params={},
        rmse=rmse,
        inlier_ratio=0.0,
        residual_p95=p95,
        confidence_class=ConfidenceClass.LOW,
        score=0.0,
        fit_source=fit_source,
        notes=note,
    )


def _rejected(t: PrimitiveType, fit_source: str, note: str) -> PrimitiveFit:
    return PrimitiveFit(
        type=t,
        params={},
        rmse=0.0,
        inlier_ratio=0.0,
        residual_p95=0.0,
        confidence_class=ConfidenceClass.REJECTED,
        score=0.0,
        fit_source=fit_source,
        notes=note,
    )


def _grade_plane(rmse: float, inliers: float, bbox_diag: float) -> ConfidenceClass:
    rel = rmse / max(bbox_diag, 1e-12)
    if rel <= PLANE_HIGH_RMSE_REL and inliers >= PLANE_HIGH_INLIER:
        return ConfidenceClass.HIGH
    if rel <= PLANE_MED_RMSE_REL and inliers >= PLANE_MED_INLIER:
        return ConfidenceClass.MEDIUM
    return ConfidenceClass.LOW


def _grade_cylinder(rmse: float, inliers: float, bbox_diag: float) -> ConfidenceClass:
    rel = rmse / max(bbox_diag, 1e-12)
    if rel <= CYL_HIGH_RMSE_REL and inliers >= CYL_HIGH_INLIER:
        return ConfidenceClass.HIGH
    if rel <= CYL_MED_RMSE_REL and inliers >= CYL_MED_INLIER:
        return ConfidenceClass.MEDIUM
    return ConfidenceClass.LOW


def _plane_score(rmse: float, inliers: float, bbox_diag: float) -> float:
    rel = rmse / max(bbox_diag, 1e-12)
    rmse_score = max(0.0, 1.0 - rel / PLANE_MED_RMSE_REL)
    return float(0.6 * inliers + 0.4 * rmse_score)


def _cylinder_score(rmse: float, inliers: float, bbox_diag: float) -> float:
    rel = rmse / max(bbox_diag, 1e-12)
    rmse_score = max(0.0, 1.0 - rel / CYL_MED_RMSE_REL)
    return float(0.5 * inliers + 0.5 * rmse_score)

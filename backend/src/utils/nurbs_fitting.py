"""Surface fitting for freeform patches using RBF and polynomial fitting."""
import numpy as np


def fit_bspline_surface(points, normals, grid_size=20):
    """Fit a smooth surface to 3D points via local PCA + polynomial/RBF fitting.

    Projects points onto a local 2D parameter space via PCA,
    then fits a polynomial z = f(u, v) in the local frame.

    Returns dict with fit parameters and quality metrics.
    """
    if len(points) < 10:
        return None

    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA for local frame
    try:
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    u_axis = vh[0]
    v_axis = vh[1]
    w_axis = vh[2]

    # Project to local coordinates
    u_coords = centered @ u_axis
    v_coords = centered @ v_axis
    w_coords = centered @ w_axis

    # Fit degree-4 polynomial for better freeform surface capture
    # w = sum of u^i * v^j terms for i+j <= 4
    u, v = u_coords, v_coords
    A = np.column_stack([
        u**4, u**3*v, u**2*v**2, u*v**3, v**4,
        u**3, u**2*v, u*v**2, v**3,
        u**2, u*v, v**2,
        u, v, np.ones(len(points))
    ])

    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(A, w_coords, rcond=None)
    except np.linalg.LinAlgError:
        return None

    w_fitted = A @ coeffs
    residual_errors = np.abs(w_coords - w_fitted)
    rmse = float(np.sqrt(np.mean(residual_errors**2)))
    max_error = float(np.max(residual_errors))

    # Curvature from the quadratic terms (indices 9,10,11 = u^2, uv, v^2)
    a_curv = coeffs[9]   # u^2 coeff
    b_curv = coeffs[10]  # uv coeff
    c_curv = coeffs[11]  # v^2 coeff
    mean_curv = float(a_curv + c_curv)
    gauss_curv = float(4*a_curv*c_curv - b_curv*b_curv)

    u_range = [float(u_coords.min()), float(u_coords.max())]
    v_range = [float(v_coords.min()), float(v_coords.max())]
    flatness = float(np.std(w_coords))

    return {
        "type": "bspline",
        "centroid": centroid.tolist(),
        "u_axis": u_axis.tolist(),
        "v_axis": v_axis.tolist(),
        "w_axis": w_axis.tolist(),
        "u_range": u_range,
        "v_range": v_range,
        "poly_coeffs": coeffs.tolist(),
        "degree": [2, 2],
        "rmse": rmse,
        "max_error": max_error,
        "flatness": flatness,
        "mean_curvature": mean_curv,
        "gaussian_curvature": gauss_curv,
        "n_points_fitted": len(points),
    }


def classify_surface_type(fit_result, points, normals):
    """Classify a surface patch using normal distribution analysis."""
    if fit_result is None:
        return "freeform"

    flatness = fit_result.get("flatness", 999)
    mean_curv = abs(fit_result.get("mean_curvature", 0))
    gauss_curv = fit_result.get("gaussian_curvature", 0)

    # Very flat
    if flatness < 0.05:
        return "planar"

    # Analyze normal distribution
    normals_centered = normals - normals.mean(axis=0)
    try:
        _, s, vh = np.linalg.svd(normals_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return "freeform"

    sv_ratio_min = s[2] / (s[0] + 1e-10)
    sv_ratio_mid = s[1] / (s[0] + 1e-10)

    # Normals lie in a plane -> cylindrical
    if sv_ratio_min < 0.15 and sv_ratio_mid > 0.3:
        return "cylindrical"

    # Normals spread equally -> spherical
    if sv_ratio_min > 0.5 and sv_ratio_mid > 0.5:
        return "spherical"

    # Gaussian curvature ~ 0 and mean curvature != 0 -> developable/cylindrical
    if abs(gauss_curv) < 0.01 and mean_curv > 0.1:
        return "cylindrical"

    # Gaussian curvature > 0 -> convex (sphere-like)
    if gauss_curv > 0.01:
        return "convex"

    # Gaussian curvature < 0 -> saddle
    if gauss_curv < -0.01:
        return "saddle"

    return "freeform"

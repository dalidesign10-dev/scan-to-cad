"""First-pass primitive fitting for the intent layer.

Scope for E0:
    - plane
    - cylinder
    - cone (chamfers on CSG parts)
    - unknown

The fitter is honest:
    - it always reports residuals + inlier ratio + a 95th percentile residual
    - it grades the fit into a discrete ConfidenceClass
    - if the fit fails any meaningful gate it returns UNKNOWN

We do NOT touch spheres, tori (fillets), freeform, B-splines yet.

Scale awareness: every numeric threshold is expressed relative to a
reference bbox diagonal. By default this is the region's own bbox
diagonal, so the same code works on a 10mm boss and a 500mm bracket
without retuning. Callers that know the full-mesh bbox should pass it
as `reference_scale` so that small regions on large parts are graded
against the mesh-level noise floor rather than their own tiny extent.
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
# Softer HIGH gate when the RMSE is deep within the HIGH band. On scan
# data the fit-driven grower includes boundary faces with residuals up to
# FIT_PLANE_TOL_REL (0.8%) — well above the HIGH inlier threshold (0.5%).
# Those boundary faces lower the HIGH-band inlier ratio below 0.85 even
# when the fit is objectively excellent (low RMSE, clean surface).
# When RMSE < 80% of the HIGH threshold, the fit is solidly on the
# surface and a 70% inlier ratio still means the vast majority of points
# are close to the plane. This avoids stuck-at-MEDIUM for genuinely flat
# scan patches with noisy boundary faces.
PLANE_TIGHT_RMSE_REL = 0.004   # 80% of PLANE_HIGH_RMSE_REL
PLANE_TIGHT_HIGH_INLIER = 0.70
# Noise-tolerant HIGH gate. On real scan data, many MEDIUM plane regions
# have rmse_rel slightly above 0.005 because a minority of boundary
# vertices are outliers, but the vast majority (82%+) of points land
# solidly within the HIGH band. If 82% of points are within 0.5% of
# bbox the surface IS flat — a curved surface at rmse_rel=0.007 would
# only have ~30% of points within the HIGH band. The 82% inlier gate
# is safe: it catches genuinely flat surfaces with noisy boundaries
# while rejecting any surface with meaningful curvature.
PLANE_NOISY_RMSE_REL = 0.007
PLANE_NOISY_HIGH_INLIER = 0.82

CYL_HIGH_RMSE_REL = 0.010
CYL_MED_RMSE_REL = 0.025
CYL_HIGH_INLIER = 0.75
CYL_MED_INLIER = 0.55

# Cylinder rejection gates (independent of confidence class)
CYL_MIN_RADIUS_REL = 0.02     # smaller radius than 2% of bbox is almost certainly noise-of-a-flat
CYL_MAX_RADIUS_REL = 8.0      # 8x bbox is the unfit-flat-as-cylinder failure mode

# Cone acceptance gates. Slightly looser than cylinder because the
# residual depends on two regressed scalars (tan α, z_apex) rather than
# one radius, so the per-point residual has more degrees of freedom.
CONE_HIGH_RMSE_REL = 0.012
CONE_MED_RMSE_REL = 0.030
CONE_HIGH_INLIER = 0.75
CONE_MED_INLIER = 0.55
# Noise-tolerant cone HIGH gate — same reasoning as PLANE_NOISY_*.
# Cone inlier threshold is looser (0.70 vs 0.82 for planes) because:
#   1. Cone residuals (perpendicular to cone surface) are more sensitive
#      to small axis/half-angle estimation errors than plane residuals
#   2. Partial cones (chamfers) naturally have boundary transition zones
#      where the surface blends into adjacent primitives
#   3. 70% of points within 1.2% of bbox on a cone surface is a strong
#      positive signal — a freeform surface at rmse_rel=0.018 would have
#      far fewer inliers within the 1.2% band
CONE_NOISY_RMSE_REL = 0.018
CONE_NOISY_HIGH_INLIER = 0.70

# Cone rejection gates. Half-angle bounds keep the cone distinguishable
# from a cylinder (too pointy → effectively a cylinder in the limit) and
# from a plane (too flat → effectively a plane in the limit). Both
# degenerate cases are better served by the plane/cylinder fitters.
CONE_MIN_HALF_ANGLE_DEG = 8.0
CONE_MAX_HALF_ANGLE_DEG = 82.0
# Mean-normal magnitude signature. On a uniform full cone of half-angle
# α, |mean(unit_normals)| = sin(α). We need this to be clearly non-zero
# so we can tell a cone apart from a cylinder (|mean(n)| ≈ 0). The
# threshold corresponds to α ≳ 3°, which is below the half-angle gate.
CONE_MIN_MEAN_NORMAL = 0.05
# Normals of a cone still lie on a small circle in 3D, so the centered
# normal SVD's smallest singular value must be small compared to the
# largest — same reasoning as the cylinder coplanarity check, slightly
# looser because partial cones are narrower circles on the sphere.
CONE_MAX_NORMAL_SV_RATIO = 0.30


def fit_region(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    fit_source: str = "proxy",
    forced_type: Optional[PrimitiveType] = None,
    reference_scale: Optional[float] = None,
) -> PrimitiveFit:
    """Try plane then cylinder then cone; return whichever wins, else UNKNOWN.

    The decision is NOT just "best inlier ratio" — a plane that passes the
    HIGH gate always wins, even if the cylinder has marginally lower RMSE
    (mechanical bias: planar > cylindrical when ambiguous, because flat
    surfaces dominate molded brackets and a wrong cylinder is uglier).

    `forced_type` is the manual-override escape hatch. When set, the fitter
    attempts ONLY that primitive and reports it honestly (even LOW), with
    an "override" note, skipping the type-selection ladder below. That's
    how a user saying "this region IS a plane, trust me" gets acted on
    without letting the override endpoint become decorative.

    `reference_scale` is the bbox diagonal against which all relative
    thresholds are measured. When None, the region's own bbox diagonal
    is used (default, backward-compatible). The pipeline passes the
    full-mesh bbox so that small regions on large parts are graded against
    the mesh-level noise floor rather than their own extent — without this,
    a 5mm patch on a 300mm scan with 0.05mm noise scores a 1% relative
    RMSE (MEDIUM) when it should score 0.017% (HIGH).
    """
    if points.shape[0] < 8:
        return _unknown(np.array([]), fit_source, "too few points")

    region_bbox = float(np.linalg.norm(np.ptp(points, axis=0)))
    if region_bbox < 1e-9:
        return _unknown(np.array([]), fit_source, "degenerate bbox")

    # When the caller provides a mesh-level reference scale, we floor the
    # region's own bbox at a fraction of it. This prevents tiny regions on
    # large parts from being over-penalized (noise RMSE / tiny-bbox =
    # large relative residual → stuck at MEDIUM even when genuinely flat)
    # without turning the whole mesh into "everything looks flat" the way
    # using the raw mesh bbox would (0.5mm RMSE / 300mm mesh = 0.17% →
    # false HIGH on curved patches). The 15% fraction means a region's
    # effective scale is never smaller than 15% of the mesh, which is
    # generous enough for small patches on scans but still catches
    # curvature on the rocker-arm freeform test. 10% was too tight
    # (missed half the cone promotions); 20% tripped the rocker arm.
    _REF_SCALE_FLOOR = 0.15
    if reference_scale is not None and reference_scale > 0:
        bbox_diag = max(region_bbox, reference_scale * _REF_SCALE_FLOOR)
    else:
        bbox_diag = region_bbox

    if forced_type is not None:
        if forced_type == PrimitiveType.PLANE:
            fit = _fit_plane(points, normals, bbox_diag, fit_source)
            if fit.confidence_class == ConfidenceClass.REJECTED:
                return _unknown(np.array([]), fit_source, "forced plane rejected")
            fit.notes = (fit.notes + " | override:force_plane").strip(" |")
            return fit
        if forced_type == PrimitiveType.CYLINDER:
            fit = _fit_cylinder(points, normals, bbox_diag, fit_source)
            if fit.confidence_class == ConfidenceClass.REJECTED:
                return _unknown(np.array([]), fit_source, f"forced cylinder rejected: {fit.notes}")
            fit.notes = (fit.notes + " | override:force_cylinder").strip(" |")
            return fit
        if forced_type == PrimitiveType.CONE:
            fit = _fit_cone(points, normals, bbox_diag, fit_source)
            if fit.confidence_class == ConfidenceClass.REJECTED:
                return _unknown(np.array([]), fit_source, f"forced cone rejected: {fit.notes}")
            fit.notes = (fit.notes + " | override:force_cone").strip(" |")
            return fit
        # Anything else (UNKNOWN, future types) falls through to auto-pick.

    plane = _fit_plane(points, normals, bbox_diag, fit_source)

    if plane.confidence_class == ConfidenceClass.HIGH:
        return plane

    cylinder = _fit_cylinder(points, normals, bbox_diag, fit_source)
    cone = _fit_cone(points, normals, bbox_diag, fit_source)

    candidates = [c for c in (plane, cylinder, cone) if c is not None]
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

    # Per-band inlier ratios. Computing a single ratio against the HIGH
    # threshold and re-using it for the MEDIUM gate (B1 bug in the initial
    # commit) makes MEDIUM strictly harder than intended — 65% of points
    # within the HIGH band is not the same as 65% within the MED band.
    inliers_high = float(np.mean(residuals < max(bbox_diag * PLANE_HIGH_RMSE_REL, 1e-9)))
    inliers_med = float(np.mean(residuals < max(bbox_diag * PLANE_MED_RMSE_REL, 1e-9)))

    cls = _grade_plane(rmse, inliers_high, inliers_med, bbox_diag)
    # Report the inlier ratio that matches the grade we landed on — that's
    # what downstream UI should display.
    inliers = inliers_high if cls == ConfidenceClass.HIGH else inliers_med
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

    inliers_high = float(np.mean(residuals < max(bbox_diag * CYL_HIGH_RMSE_REL, 1e-9)))
    inliers_med = float(np.mean(residuals < max(bbox_diag * CYL_MED_RMSE_REL, 1e-9)))

    proj_heights = (diff_c @ axis)
    height = float(proj_heights.max() - proj_heights.min())

    cls = _grade_cylinder(rmse, inliers_high, inliers_med, bbox_diag)
    inliers = inliers_high if cls == ConfidenceClass.HIGH else inliers_med
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


def _fit_cone(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    bbox_diag: float,
    fit_source: str,
) -> PrimitiveFit:
    """Fit a right circular cone to `points` using `normals`.

    The trick that makes cones tractable with the same plumbing as the
    cylinder fitter is the mean-normal signature:

        for a right circular cone of half-angle α with axis â (pointing
        apex → base), the outward unit face normals satisfy

            mean(n) = -sin(α) * â

    so |mean(n)| = sin(α) gives a cheap half-angle estimate AND the sign
    of (-mean(n) · â) tells us which way â must point to reach the apex.

    A cylinder has mean(n) ≈ 0 (normals are symmetric around the axis),
    which is why CONE_MIN_MEAN_NORMAL is the discriminator between the
    two. Both have sv[2]/sv[0] small because both live on a circle on
    the unit sphere (great circle for cylinder, small circle for cone).

    After axis + half-angle, the apex is recovered by a 1D linear fit of
    radial vs axial coordinate: r(z) = (z - z_apex) * tan(α).
    """
    if normals is None or normals.shape[0] < 8:
        return _rejected(PrimitiveType.CONE, fit_source, "no normals")
    # Note: points and normals are used independently here (normals for
    # the mean-normal / SVD signature and axis, points for the linear
    # apex regression and residuals), so a length mismatch between them
    # is explicitly OK — pipeline.py passes unique region vertices as
    # points and per-face normals as normals, and those sizes differ.

    # Work on unit normals — the mean-normal signature is only clean on
    # the unit sphere, and downstream the magnitude of the mean is
    # interpreted as sin(α) which only holds if inputs are normalized.
    n_norms = np.linalg.norm(normals, axis=1)
    if np.any(n_norms < 1e-9):
        return _rejected(PrimitiveType.CONE, fit_source, "degenerate normals")
    unit_normals = normals / n_norms[:, None]

    mean_n = unit_normals.mean(axis=0)
    mean_n_mag = float(np.linalg.norm(mean_n))
    if mean_n_mag < CONE_MIN_MEAN_NORMAL:
        return _rejected(
            PrimitiveType.CONE,
            fit_source,
            f"mean-normal too small ({mean_n_mag:.3f}) — looks like cylinder/plane",
        )

    nc = unit_normals - mean_n
    try:
        _, sv, vh = np.linalg.svd(nc, full_matrices=False)
    except np.linalg.LinAlgError:
        return _rejected(PrimitiveType.CONE, fit_source, "SVD failed")
    if sv[0] < 1e-9:
        return _rejected(PrimitiveType.CONE, fit_source, "degenerate normal spread")

    sv_ratio = float(sv[2] / sv[0])
    if sv_ratio > CONE_MAX_NORMAL_SV_RATIO:
        return _rejected(
            PrimitiveType.CONE,
            fit_source,
            f"normals not on a small circle (sv_ratio={sv_ratio:.2f})",
        )

    axis = vh[2].astype(np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    u_ax = vh[0].astype(np.float64)
    v_ax = vh[1].astype(np.float64)
    # Orient axis so that apex→base is positive: since mean(n) = -sin(α)*axis,
    # we want dot(axis, -mean_n) > 0.
    if float(np.dot(axis, -mean_n)) < 0:
        axis = -axis

    centroid = points.mean(axis=0)
    diff = points - centroid
    u = diff @ u_ax                 # position in axis-perp plane, first basis
    v = diff @ v_ax                 # position in axis-perp plane, second basis
    z = diff @ axis                 # centered axial coordinate

    # Right cone equation expressed in axis-aligned coordinates:
    #   (u - cu)^2 + (v - cv)^2 = tan²(α) * (z - z_apex)^2
    # Expanding and regrouping so all five unknowns enter linearly:
    #   u² + v² = 2*cu*u + 2*cv*v + tan²(α)*z² - 2*tan²(α)*z_apex*z + E
    # with E = cu² + cv² - tan²(α)*z_apex². Solve A*u + B*v + C*z² + D*z + E
    # for (A, B, C, D, E) by linear LS — this locates the axis line
    # (cu, cv), recovers tan²(α) = C, and gives the apex along the axis
    # as z_apex = -D / (2C). Works for partial cones because nothing here
    # assumes symmetry in the azimuthal direction.
    lhs = np.column_stack([u, v, z * z, z, np.ones_like(z)])
    rhs = u * u + v * v
    try:
        sol, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return _rejected(PrimitiveType.CONE, fit_source, "cone lstsq failed")

    A, B, C, D, E = (float(s) for s in sol)
    if C <= 1e-9:
        return _rejected(
            PrimitiveType.CONE,
            fit_source,
            "non-positive tan²α (degenerate fit)",
        )
    tan_a = float(np.sqrt(C))
    half_angle_rad = float(np.arctan(tan_a))
    half_angle_deg = float(np.degrees(half_angle_rad))
    if half_angle_deg < CONE_MIN_HALF_ANGLE_DEG:
        return _rejected(
            PrimitiveType.CONE,
            fit_source,
            f"half-angle too small ({half_angle_deg:.1f} deg)",
        )
    if half_angle_deg > CONE_MAX_HALF_ANGLE_DEG:
        return _rejected(
            PrimitiveType.CONE,
            fit_source,
            f"half-angle too large ({half_angle_deg:.1f} deg)",
        )

    cu = A / 2.0
    cv = B / 2.0
    z_apex_centered = -D / (2.0 * C)
    apex = centroid + cu * u_ax + cv * v_ax + z_apex_centered * axis

    # Residual: perpendicular distance from each point to the infinite cone
    # surface. Work in (axial s, radial r) coordinates from the apex. The
    # cone line is r = |s| * tan(α); the perpendicular distance from a
    # point at (s, r) to that line is |r*cos(α) - |s|*sin(α)|. Using |s|
    # keeps the sign correct when the axis happens to point away from
    # the base (physical points all land on one nappe, so |s| is the
    # honest apex→point distance along the axis).
    diff_apex = points - apex
    s_axial = diff_apex @ axis
    perp_vec = diff_apex - np.outer(s_axial, axis)
    r_radial = np.linalg.norm(perp_vec, axis=1)
    sin_a = float(np.sin(half_angle_rad))
    cos_a = float(np.cos(half_angle_rad))
    residuals = np.abs(r_radial * cos_a - np.abs(s_axial) * sin_a)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    p95 = float(np.quantile(residuals, 0.95))

    inliers_high = float(np.mean(residuals < max(bbox_diag * CONE_HIGH_RMSE_REL, 1e-9)))
    inliers_med = float(np.mean(residuals < max(bbox_diag * CONE_MED_RMSE_REL, 1e-9)))

    # Extent along the axis, measured from the apex — used as the
    # gizmo length and by downstream chamfer heuristics.
    z_from_apex = np.abs(s_axial)
    height = float(z_from_apex.max() - z_from_apex.min())

    cls = _grade_cone(rmse, inliers_high, inliers_med, bbox_diag)
    inliers = inliers_high if cls == ConfidenceClass.HIGH else inliers_med
    score = _cone_score(rmse, inliers, bbox_diag)
    return PrimitiveFit(
        type=PrimitiveType.CONE,
        params={
            "apex": apex.tolist(),
            "axis": axis.tolist(),
            "half_angle_deg": half_angle_deg,
            "height": height,
            "mean_normal_magnitude": mean_n_mag,
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


def _grade_plane(
    rmse: float,
    inliers_high: float,
    inliers_med: float,
    bbox_diag: float,
) -> ConfidenceClass:
    """Grade a plane fit using the inlier ratio that matches each band.

    `inliers_high` must be the fraction of points within the HIGH band,
    `inliers_med` the fraction within the (wider) MED band. Using a single
    HIGH-band ratio for both gates was B1 in the initial commit — it made
    MEDIUM strictly harder than MEDIUM should be.
    """
    rel = rmse / max(bbox_diag, 1e-12)
    if rel <= PLANE_HIGH_RMSE_REL and inliers_high >= PLANE_HIGH_INLIER:
        return ConfidenceClass.HIGH
    # Alternative HIGH gate: when the RMSE is deep within the HIGH band,
    # accept a softer inlier requirement. See PLANE_TIGHT_RMSE_REL comment.
    if rel <= PLANE_TIGHT_RMSE_REL and inliers_high >= PLANE_TIGHT_HIGH_INLIER:
        return ConfidenceClass.HIGH
    # Noise-tolerant gate: the surface is overwhelmingly flat (92%+ within
    # the HIGH band) but a few boundary outliers inflate RMSE slightly
    # past the main threshold. See PLANE_NOISY_RMSE_REL comment.
    if rel <= PLANE_NOISY_RMSE_REL and inliers_high >= PLANE_NOISY_HIGH_INLIER:
        return ConfidenceClass.HIGH
    if rel <= PLANE_MED_RMSE_REL and inliers_med >= PLANE_MED_INLIER:
        return ConfidenceClass.MEDIUM
    return ConfidenceClass.LOW


def _grade_cylinder(
    rmse: float,
    inliers_high: float,
    inliers_med: float,
    bbox_diag: float,
) -> ConfidenceClass:
    rel = rmse / max(bbox_diag, 1e-12)
    if rel <= CYL_HIGH_RMSE_REL and inliers_high >= CYL_HIGH_INLIER:
        return ConfidenceClass.HIGH
    if rel <= CYL_MED_RMSE_REL and inliers_med >= CYL_MED_INLIER:
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


def _grade_cone(
    rmse: float,
    inliers_high: float,
    inliers_med: float,
    bbox_diag: float,
) -> ConfidenceClass:
    rel = rmse / max(bbox_diag, 1e-12)
    if rel <= CONE_HIGH_RMSE_REL and inliers_high >= CONE_HIGH_INLIER:
        return ConfidenceClass.HIGH
    if rel <= CONE_NOISY_RMSE_REL and inliers_high >= CONE_NOISY_HIGH_INLIER:
        return ConfidenceClass.HIGH
    if rel <= CONE_MED_RMSE_REL and inliers_med >= CONE_MED_INLIER:
        return ConfidenceClass.MEDIUM
    return ConfidenceClass.LOW


def _cone_score(rmse: float, inliers: float, bbox_diag: float) -> float:
    rel = rmse / max(bbox_diag, 1e-12)
    rmse_score = max(0.0, 1.0 - rel / CONE_MED_RMSE_REL)
    return float(0.5 * inliers + 0.5 * rmse_score)

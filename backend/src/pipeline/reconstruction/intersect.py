"""Phase E0 — family-level analytic intersections.

Every SurfaceFamily is a canonical analytic surface; two surfaces that
meet along a sharp edge of the part imply a real analytic curve (a
line for plane/plane, an ellipse for plane/cylinder, etc.). This
module produces that set of curves.

Design rules:

  1. Family-level, not region-level. Two coplanar pads share a family,
     so they never produce a duplicate intersection — one plane pair
     yields one line, regardless of how many physically separate
     regions sit on it.

  2. Adjacency-pruned. We do NOT compute every O(n²) family pair; we
     walk ``state.boundaries`` and only intersect family pairs whose
     member regions actually share a SHARP boundary. That trims a
     ~3600-pair phantom set on a real scan down to the ~50 edges the
     user can see on the mesh.

  3. JSON-shaped output. Each intersection is a plain dict matching
     the frontend's IntentFamilyEdge interface — keeps the overlay
     shipping path dead simple.

  4. Supported intersection types: plane/plane (line), plane/cylinder
     (circle, ellipse, or pair of lines), plane/cone (conic section),
     and cylinder/cylinder (numerical space curve).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .state import (
    Boundary,
    PrimitiveType,
    Region,
    ReconstructionState,
    SurfaceFamily,
)


# ─────────────────────────────────────────────────────────────────────────
# Clipping helpers
# ─────────────────────────────────────────────────────────────────────────

def _clip_line_to_aabb(
    p0: np.ndarray,
    direction: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Liang-Barsky line-vs-AABB clip.

    ``p0`` is any point on the infinite line, ``direction`` is a unit
    vector. Returns the two clipped endpoints, or None if the line
    misses the box (or is degenerate in every axis and out of range).
    """
    t_min, t_max = -np.inf, np.inf
    for ax in range(3):
        if abs(direction[ax]) < 1e-9:
            if p0[ax] < bbox_min[ax] or p0[ax] > bbox_max[ax]:
                return None
            continue
        t1 = (bbox_min[ax] - p0[ax]) / direction[ax]
        t2 = (bbox_max[ax] - p0[ax]) / direction[ax]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
        if t2 < t_max:
            t_max = t2
        if t_min > t_max:
            return None
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return None
    return p0 + t_min * direction, p0 + t_max * direction


# ─────────────────────────────────────────────────────────────────────────
# Family pair discovery
# ─────────────────────────────────────────────────────────────────────────

def _adjacent_family_pairs(
    state: ReconstructionState,
) -> Dict[Tuple[int, int], List[int]]:
    """Walk state.boundaries and collect family pairs that share at least
    one sharp boundary.

    Returns a dict keyed by ``(fa, fb)`` with ``fa < fb``, whose values
    are the boundary indices that map to that family pair (handy for
    downstream provenance / debugging).

    Two regions on the SAME family (coplanar pads) are skipped — there
    is no intersection to compute, they are literally the same surface.
    Non-HIGH regions (family_id < 0) are skipped because they have no
    canonical primitive.
    """
    pairs: Dict[Tuple[int, int], List[int]] = {}
    for bi, b in enumerate(state.boundaries):
        if not b.sharp:
            continue
        ra = state.regions.get(b.region_a)
        rb = state.regions.get(b.region_b)
        if ra is None or rb is None:
            continue
        fa = ra.surface_family_id
        fb = rb.surface_family_id
        if fa < 0 or fb < 0:
            continue
        # Both regions must live in a SurfaceFamily object — that
        # guarantees HIGH + known primitive type.
        if fa not in state.surface_families or fb not in state.surface_families:
            continue
        if fa == fb:
            continue
        key = (fa, fb) if fa < fb else (fb, fa)
        pairs.setdefault(key, []).append(bi)
    return pairs


# ─────────────────────────────────────────────────────────────────────────
# Individual intersection cases
# ─────────────────────────────────────────────────────────────────────────

def _plane_plane_edge(
    fa: SurfaceFamily,
    fb: SurfaceFamily,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Optional[List[List[float]]]:
    """Plane × plane → line segment, clipped to the mesh AABB."""
    pa = fa.canonical_params
    pb = fb.canonical_params
    try:
        n1 = np.asarray(pa["normal"], dtype=float)
        n2 = np.asarray(pb["normal"], dtype=float)
        d1 = float(pa["d"])
        d2 = float(pb["d"])
    except (KeyError, TypeError, ValueError):
        return None
    # Parallel check: |n1 × n2| ≈ 0 means parallel planes.
    direction = np.cross(n1, n2)
    denom = float(np.linalg.norm(direction))
    if denom < 1e-6:
        return None
    direction /= denom
    # Point on the line: solve n1·p = -d1, n2·p = -d2, direction·p = 0.
    # (The direction·p = 0 row picks the foot of the perpendicular from
    # the origin onto the line, which keeps p0 numerically stable.)
    A = np.stack([n1, n2, direction])
    b = np.array([-d1, -d2, 0.0])
    try:
        p0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    clipped = _clip_line_to_aabb(p0, direction, bbox_min, bbox_max)
    if clipped is None:
        return None
    a, c = clipped
    return [a.tolist(), c.tolist()]


def _filter_points_aabb(
    pts: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Optional[np.ndarray]:
    """Keep only rows of *pts* (N, 3) that fall inside the padded AABB.

    Returns the filtered array, or None if nothing survives.
    """
    if pts.size == 0:
        return None
    mask = np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)
    out = pts[mask]
    return out if out.shape[0] >= 2 else None


def _build_orthonormal_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, w) where w = axis (unit) and u, v span the normal plane."""
    w = axis / np.linalg.norm(axis)
    # Pick a seed vector not parallel to w.
    seed = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, seed)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v, w


# ─── Plane × Cylinder ────────────────────────────────────────────────────

def _plane_cylinder_edge(
    fa_plane: SurfaceFamily,
    fa_cyl: SurfaceFamily,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_samples: int = 64,
) -> Optional[List[List[float]]]:
    """Plane × cylinder → circle, ellipse, or pair of lines, clipped to AABB."""
    pp = fa_plane.canonical_params
    cp = fa_cyl.canonical_params
    try:
        plane_n = np.asarray(pp["normal"], dtype=float)
        plane_d = float(pp["d"])
        cyl_axis = np.asarray(cp["axis"], dtype=float)
        cyl_center = np.asarray(cp["center"], dtype=float)
        cyl_radius = float(cp["radius"])
    except (KeyError, TypeError, ValueError):
        return None

    plane_n = plane_n / np.linalg.norm(plane_n)
    cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)

    cos_theta = abs(float(np.dot(plane_n, cyl_axis)))

    # --- Case 1: axis perpendicular to plane (cos_theta ≈ 1) → circle ---
    if cos_theta > 1.0 - 1e-6:
        # The intersection is a circle of radius R in the plane.
        # Find the point where the cylinder axis meets the plane.
        # Plane eq: n·p + d = 0.  Axis line: cyl_center + t * cyl_axis.
        denom = float(np.dot(plane_n, cyl_axis))
        if abs(denom) < 1e-12:
            return None
        t = -(float(np.dot(plane_n, cyl_center)) + plane_d) / denom
        circle_center = cyl_center + t * cyl_axis

        # Sample the circle.
        u, v, _w = _build_orthonormal_frame(cyl_axis)
        theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        pts = (
            circle_center[None, :]
            + cyl_radius * np.cos(theta)[:, None] * u[None, :]
            + cyl_radius * np.sin(theta)[:, None] * v[None, :]
        )
        # Close the loop.
        pts = np.vstack([pts, pts[0:1]])
        filtered = _filter_points_aabb(pts, bbox_min, bbox_max)
        if filtered is None:
            return None
        return filtered.tolist()

    # --- Case 2: axis lies in the plane (cos_theta ≈ 0) → two parallel lines ---
    if cos_theta < 1e-6:
        # Two lines parallel to cyl_axis, offset by ±R in the direction
        # that is perpendicular to both cyl_axis and plane_n.
        perp = np.cross(cyl_axis, plane_n)
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-12:
            return None
        perp /= perp_norm
        lines: List[List[float]] = []
        for sign in (+1.0, -1.0):
            p0 = cyl_center + sign * cyl_radius * perp
            # Project p0 onto the plane to remove any small offset.
            p0 -= (float(np.dot(plane_n, p0)) + plane_d) * plane_n
            clipped = _clip_line_to_aabb(p0, cyl_axis, bbox_min, bbox_max)
            if clipped is not None:
                lines.append(clipped[0].tolist())
                lines.append(clipped[1].tolist())
        return lines if len(lines) >= 2 else None

    # --- Case 3: general oblique intersection → ellipse ---
    # Work in cylinder-local frame: origin = cyl_center, Z = cyl_axis.
    u, v, w = _build_orthonormal_frame(cyl_axis)
    # Rotation matrix world→local: rows are u, v, w.
    R = np.stack([u, v, w])  # (3, 3)

    # Plane normal and offset in local frame.
    n_loc = R @ plane_n  # (3,)
    # plane eq in world: plane_n · p + plane_d = 0
    # p_world = R^T p_loc + cyl_center
    # plane_n · (R^T p_loc + cyl_center) + plane_d = 0
    # n_loc · p_loc + (plane_n · cyl_center + plane_d) = 0
    d_loc = float(np.dot(plane_n, cyl_center)) + plane_d

    # In local frame, cylinder surface: x = R cos θ, y = R sin θ, z free.
    # Plane: n_loc[0]*x + n_loc[1]*y + n_loc[2]*z + d_loc = 0
    # Solve for z(θ):
    #   z(θ) = -(n_loc[0]*R*cos θ + n_loc[1]*R*sin θ + d_loc) / n_loc[2]
    if abs(n_loc[2]) < 1e-12:
        # Degenerate: plane is parallel to cylinder axis in local frame.
        # This should have been caught by the cos_theta ≈ 0 branch.
        return None

    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    x_loc = cyl_radius * np.cos(theta)
    y_loc = cyl_radius * np.sin(theta)
    z_loc = -(n_loc[0] * x_loc + n_loc[1] * y_loc + d_loc) / n_loc[2]

    # Transform back to world.
    pts_loc = np.stack([x_loc, y_loc, z_loc], axis=1)  # (N, 3)
    pts_world = (R.T @ pts_loc.T).T + cyl_center  # (N, 3)
    # Close the loop.
    pts_world = np.vstack([pts_world, pts_world[0:1]])

    filtered = _filter_points_aabb(pts_world, bbox_min, bbox_max)
    if filtered is None:
        return None
    return filtered.tolist()


# ─── Plane × Cone ────────────────────────────────────────────────────────

def _plane_cone_edge(
    fa_plane: SurfaceFamily,
    fa_cone: SurfaceFamily,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_samples: int = 64,
) -> Optional[List[List[float]]]:
    """Plane × cone → conic section (circle, ellipse, parabola, hyperbola, or line pair).

    Returns a sampled polyline clipped to the AABB, or None.
    """
    pp = fa_plane.canonical_params
    cp = fa_cone.canonical_params
    try:
        plane_n = np.asarray(pp["normal"], dtype=float)
        plane_d = float(pp["d"])
        apex = np.asarray(cp["apex"], dtype=float)
        cone_axis = np.asarray(cp["axis"], dtype=float)
        half_angle_deg = float(cp["half_angle_deg"])
    except (KeyError, TypeError, ValueError):
        return None

    plane_n = plane_n / np.linalg.norm(plane_n)
    cone_axis = cone_axis / np.linalg.norm(cone_axis)
    alpha = np.radians(half_angle_deg)
    tan_a = np.tan(alpha)

    if tan_a < 1e-12:
        return None  # Degenerate cone (zero half-angle → line)

    # Work in cone-local frame: origin = apex, Z = cone_axis.
    u, v, w = _build_orthonormal_frame(cone_axis)
    R = np.stack([u, v, w])  # world→local

    # Plane in local frame.
    n_loc = R @ plane_n
    # plane eq world: plane_n · p + plane_d = 0
    # p_world = R^T p_loc + apex
    # n_loc · p_loc + (plane_n · apex + plane_d) = 0
    d_loc = float(np.dot(plane_n, apex)) + plane_d

    # Cone surface in local frame (both nappes): x²+y² = z²·tan²(α)
    # Parametrize by θ: x = r cosθ, y = r sinθ, r = |z|·tan(α)
    # We allow z > 0 and z < 0 (both nappes).
    #
    # Plane: n_loc[0]*x + n_loc[1]*y + n_loc[2]*z + d_loc = 0
    # Substitute x = z·tan(α)·cosθ, y = z·tan(α)·sinθ  (for z > 0 nappe):
    #   z * (n_loc[0]*tan(α)*cosθ + n_loc[1]*tan(α)*sinθ + n_loc[2]) + d_loc = 0
    #   z = -d_loc / (n_loc[0]*tan(α)*cosθ + n_loc[1]*tan(α)*sinθ + n_loc[2])

    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    denom = n_loc[0] * tan_a * cos_t + n_loc[1] * tan_a * sin_t + n_loc[2]

    all_points = []
    for nappe_sign in (+1.0, -1.0):
        # For the negative nappe, x = -z·tan(α)·cosθ etc. with z < 0,
        # equivalently z_nappe = -d_loc / denom_nappe.
        if nappe_sign > 0:
            d_arr = denom
        else:
            # Negative nappe: x = |z|·tan(α)·cosθ, y = |z|·tan(α)·sinθ, z < 0
            # so z = -|z|, i.e. x = -z·tan(α)·cosθ ...
            d_arr = -n_loc[0] * tan_a * cos_t - n_loc[1] * tan_a * sin_t + n_loc[2]

        # Avoid division by zero / near-zero.
        valid = np.abs(d_arr) > 1e-12
        if not np.any(valid):
            continue

        z_loc = np.full_like(theta, np.nan)
        z_loc[valid] = -d_loc / d_arr[valid]

        # Only keep points on the correct nappe.
        if nappe_sign > 0:
            nappe_ok = z_loc >= 0
        else:
            nappe_ok = z_loc <= 0

        ok = valid & nappe_ok & np.isfinite(z_loc)
        if not np.any(ok):
            continue

        r_loc = np.abs(z_loc) * tan_a
        x_loc = r_loc * cos_t
        y_loc = r_loc * sin_t

        pts_loc = np.stack([x_loc[ok], y_loc[ok], z_loc[ok]], axis=1)
        pts_world = (R.T @ pts_loc.T).T + apex
        all_points.append(pts_world)

    if not all_points:
        return None

    pts = np.vstack(all_points)

    # Handle degenerate case: plane through apex produces a point or a
    # pair of lines.  If abs(d_loc) is tiny, many z values will cluster
    # near zero (the apex).  Check and return early.
    if abs(d_loc) < 1e-9:
        # Plane passes through apex.  Intersection is 0, 1, or 2 lines
        # through the apex.  Build them from the directions where the
        # plane meets the cone.
        # Direction on cone: d = tanα (u cosθ + v sinθ) + w
        # Plane: n_loc · d = 0 → n_loc[0]*tanα*cosθ + n_loc[1]*tanα*sinθ + n_loc[2] = 0
        # → tanα * (n_loc[0] cosθ + n_loc[1] sinθ) = -n_loc[2]
        # Let A = tanα * n_loc[0], B = tanα * n_loc[1], C = -n_loc[2]
        # A cosθ + B sinθ = C → solve for θ.
        A = tan_a * n_loc[0]
        B = tan_a * n_loc[1]
        C = -n_loc[2]
        mag = np.hypot(A, B)
        if mag < 1e-12:
            return None
        ratio = C / mag
        if abs(ratio) > 1.0 + 1e-9:
            return None
        ratio = np.clip(ratio, -1.0, 1.0)
        base_angle = np.arctan2(B, A)
        delta = np.arccos(ratio)
        lines: List[List[float]] = []
        for theta_sol in [base_angle + delta, base_angle - delta]:
            direction_loc = np.array([
                tan_a * np.cos(theta_sol),
                tan_a * np.sin(theta_sol),
                1.0,
            ])
            direction_world = R.T @ direction_loc
            direction_world /= np.linalg.norm(direction_world)
            clipped = _clip_line_to_aabb(apex, direction_world, bbox_min, bbox_max)
            if clipped is not None:
                lines.append(clipped[0].tolist())
                lines.append(clipped[1].tolist())
        return lines if len(lines) >= 2 else None

    filtered = _filter_points_aabb(pts, bbox_min, bbox_max)
    if filtered is None:
        return None
    return filtered.tolist()


# ─── Cylinder × Cylinder ─────────────────────────────────────────────────

def _cylinder_cylinder_edge(
    fa_a: SurfaceFamily,
    fa_b: SurfaceFamily,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_samples: int = 128,
) -> Optional[List[List[float]]]:
    """Cylinder × cylinder → space curve, found numerically.

    Samples cylinder A surface, evaluates signed distance to cylinder B,
    finds zero-crossings via bisection, then sorts into a polyline.
    Returns the polyline clipped to AABB, or None.
    """
    pa = fa_a.canonical_params
    pb = fa_b.canonical_params
    try:
        axis_a = np.asarray(pa["axis"], dtype=float)
        center_a = np.asarray(pa["center"], dtype=float)
        radius_a = float(pa["radius"])
        axis_b = np.asarray(pb["axis"], dtype=float)
        center_b = np.asarray(pb["center"], dtype=float)
        radius_b = float(pb["radius"])
    except (KeyError, TypeError, ValueError):
        return None

    axis_a = axis_a / np.linalg.norm(axis_a)
    axis_b = axis_b / np.linalg.norm(axis_b)

    # Determine a reasonable t-range for sampling along cylinder A's axis.
    # Use the AABB diagonal as a conservative bound.
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    t_half = diag * 0.6

    def _signed_dist_to_cyl_b(pts: np.ndarray) -> np.ndarray:
        """Signed distance of (N,3) points to the surface of cylinder B.

        Positive outside, negative inside.
        """
        diff = pts - center_b[None, :]
        proj = np.dot(diff, axis_b)[:, None] * axis_b[None, :]
        radial = diff - proj
        radial_dist = np.linalg.norm(radial, axis=1)
        return radial_dist - radius_b

    # Build an orthonormal frame for cylinder A.
    u_a, v_a, w_a = _build_orthonormal_frame(axis_a)

    n_theta = n_samples
    n_t = n_samples
    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    t_vals = np.linspace(-t_half, t_half, n_t)

    # Evaluate signed distance on a (theta, t) grid.
    # Shape: (n_theta, n_t)
    # For each theta, sweep along t and find sign changes.
    zero_pts = []

    for i_th, th in enumerate(theta_vals):
        # Points on cylinder A at this theta for all t values.
        radial_dir = np.cos(th) * u_a + np.sin(th) * v_a
        line_pts = center_a[None, :] + radius_a * radial_dir[None, :] + t_vals[:, None] * w_a[None, :]
        sd = _signed_dist_to_cyl_b(line_pts)  # (n_t,)

        # Find sign changes.
        signs = np.sign(sd)
        for j in range(len(signs) - 1):
            if signs[j] == 0:
                zero_pts.append(line_pts[j])
            elif signs[j + 1] == 0:
                continue  # Will be caught at next j.
            elif signs[j] != signs[j + 1]:
                # Bisect to refine.
                t_lo, t_hi = t_vals[j], t_vals[j + 1]
                for _ in range(20):
                    t_mid = 0.5 * (t_lo + t_hi)
                    p_mid = center_a + radius_a * radial_dir + t_mid * w_a
                    sd_mid = float(np.linalg.norm(
                        (p_mid - center_b) - np.dot(p_mid - center_b, axis_b) * axis_b
                    )) - radius_b
                    if abs(sd_mid) < 1e-9:
                        break
                    if np.sign(sd_mid) == signs[j]:
                        t_lo = t_mid
                    else:
                        t_hi = t_mid
                p_refined = center_a + radius_a * radial_dir + 0.5 * (t_lo + t_hi) * w_a
                zero_pts.append(p_refined)

    if len(zero_pts) < 2:
        return None

    pts = np.array(zero_pts)

    # Sort into a polyline by nearest-neighbor greedy walk.
    ordered = [0]
    remaining = set(range(1, len(pts)))
    while remaining:
        last = ordered[-1]
        dists = np.linalg.norm(pts[list(remaining)] - pts[last], axis=1)
        rem_list = list(remaining)
        nearest_idx = rem_list[int(np.argmin(dists))]
        ordered.append(nearest_idx)
        remaining.remove(nearest_idx)
    pts = pts[ordered]

    filtered = _filter_points_aabb(pts, bbox_min, bbox_max)
    if filtered is None:
        return None
    return filtered.tolist()


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def compute_family_edges(
    state: ReconstructionState,
    mesh_vertices: Optional[np.ndarray] = None,
) -> List[dict]:
    """Build the family-level intersection edges for a ReconstructionState.

    Requires ``state.surface_families`` to be populated. Supply
    ``mesh_vertices`` when available — we use them to derive the AABB
    used for plane-plane clipping. If absent we fall back to a bbox
    derived from the proxy mesh (always available).

    Returns a list of JSON-shaped edge dicts. Current schema (subject
    to extension, not removal):

        {
          "family_a": int,
          "family_b": int,
          "type_a":   "plane" | ...,
          "type_b":   "plane" | ...,
          "kind":     "plane_plane",
          "points":   [[x,y,z], [x,y,z], ...],
          "n_points": int,
          "n_supporting_boundaries": int,   # for provenance
        }

    The ``points`` list is a polyline (length ≥ 2). For plane/plane it
    is exactly the two clipped line endpoints.
    """
    if not state.surface_families:
        return []

    if mesh_vertices is not None and mesh_vertices.size > 0:
        verts = np.asarray(mesh_vertices, dtype=float)
    else:
        verts = np.asarray(state.proxy.vertices, dtype=float)
    if verts.size == 0:
        return []
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    pad = 0.02 * diag
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad

    pairs = _adjacent_family_pairs(state)
    if not pairs:
        return []

    edges: List[dict] = []
    for (fa_id, fb_id), boundary_ids in pairs.items():
        fa = state.surface_families[fa_id]
        fb = state.surface_families[fb_id]

        type_a = fa.type.value
        type_b = fb.type.value

        points: Optional[List[List[float]]] = None
        kind: Optional[str] = None

        if type_a == "plane" and type_b == "plane":
            points = _plane_plane_edge(fa, fb, bbox_min, bbox_max)
            kind = "plane_plane"
        elif ("plane" in (type_a, type_b)) and ("cylinder" in (type_a, type_b)):
            fa_p, fa_c = (fa, fb) if type_a == "plane" else (fb, fa)
            points = _plane_cylinder_edge(fa_p, fa_c, bbox_min, bbox_max)
            kind = "plane_cylinder"
        elif ("plane" in (type_a, type_b)) and ("cone" in (type_a, type_b)):
            fa_p, fa_co = (fa, fb) if type_a == "plane" else (fb, fa)
            points = _plane_cone_edge(fa_p, fa_co, bbox_min, bbox_max)
            kind = "plane_cone"
        elif type_a == "cylinder" and type_b == "cylinder":
            points = _cylinder_cylinder_edge(fa, fb, bbox_min, bbox_max)
            kind = "cylinder_cylinder"
        else:
            continue

        if points is None:
            continue

        edges.append({
            "family_a": int(fa_id),
            "family_b": int(fb_id),
            "type_a": type_a,
            "type_b": type_b,
            "kind": kind,
            "points": points,
            "n_points": len(points),
            "n_supporting_boundaries": len(boundary_ids),
        })

    return edges

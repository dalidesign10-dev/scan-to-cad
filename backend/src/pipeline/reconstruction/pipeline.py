"""Phase E0 — top-level intent reconstruction entrypoint.

Wires the proxy mesh, boundary detector, region grower and fitter together
into a single ReconstructionState attached to the FastAPI session.

The function is intentionally linear and reads top-to-bottom:

    cleaned mesh
       │
       ▼
    proxy mesh ─────────────────────────────────┐
       │                                        │
       ▼                                        │
    boundary signals (dihedral + jump + curv)   │
       │                                        │
       ▼                                        │
    region grow + region adjacency graph        │
       │                                        │
       ▼                                        │
    proxy fits (plane / cylinder / unknown)     │
       │                                        │
       ▼                                        │
    label transfer back to full-res ◄───────────┘
       │
       ▼
    full-res refit, preserving fit_proxy diagnostics

Nothing here mutates Phase A/B/C state.
"""

import time
from typing import Optional
import numpy as np

from .state import (
    ReconstructionState,
    Region,
    PrimitiveFit,
    PrimitiveType,
    ConfidenceClass,
)
from .proxy_mesh import build_proxy_mesh, transfer_labels_to_full
from .boundary import compute_boundary_signals
from .regions import grow_regions, build_region_graph
from .fitting import fit_region


def run_intent_segmentation(
    params: dict,
    progress_callback=None,
    session=None,
) -> dict:
    """Build a fresh ReconstructionState on `session["recon_state"]`.

    Params (all optional):
        target_proxy_faces  int    default 30000
        min_region_faces    int    default 12
        growth_mode         str    default "dihedral" — {"dihedral","fit_driven"}
        force_rebuild       bool   default True

    growth_mode="dihedral" is the clean-CAD grower (hybrid boundary + soft
    normal gate). growth_mode="fit_driven" is the RANSAC-style primitive
    grower for scan data where dihedral boundaries don't form closed loops.
    """
    if session is None:
        raise ValueError("session required")
    full_mesh = session.get("preprocessed") or session.get("mesh")
    if full_mesh is None:
        raise ValueError("Run cleanup (Phase A) before intent reconstruction")

    target_proxy = int(params.get("target_proxy_faces", 40000))
    min_region_faces = int(params.get("min_region_faces", 12))
    growth_mode = str(params.get("growth_mode", "dihedral"))
    if growth_mode not in ("dihedral", "fit_driven"):
        raise ValueError(f"unknown growth_mode: {growth_mode!r}")

    t0 = time.time()
    if progress_callback:
        progress_callback("intent", 2, "Building proxy mesh...")

    proxy = build_proxy_mesh(
        full_mesh,
        target_face_count=target_proxy,
        progress_callback=lambda *a, **k: None,
    )

    if progress_callback:
        progress_callback("intent", 20, "Computing hybrid boundary signals...")

    signals = compute_boundary_signals(proxy, progress_callback=lambda *a, **k: None)

    if progress_callback:
        progress_callback("intent", 40, "Growing regions...")

    proxy_labels = grow_regions(
        proxy,
        signals,
        min_region_face_count=min_region_faces,
        progress_callback=lambda *a, **k: None,
        mode=growth_mode,
    )

    if progress_callback:
        progress_callback("intent", 60, "Building region graph...")

    boundaries = build_region_graph(proxy_labels, signals)

    if progress_callback:
        progress_callback("intent", 65, "Transferring labels to full mesh...")

    full_labels = transfer_labels_to_full(proxy_labels, proxy)

    if progress_callback:
        progress_callback("intent", 75, "Proxy fits per region...")

    n_regions = int(proxy_labels.max()) + 1
    full_vertices = np.asarray(full_mesh.vertices, dtype=np.float64)
    full_faces = np.asarray(full_mesh.faces, dtype=np.int64)

    # Mesh-level reference scale: the full-mesh bbox diagonal. Passed to
    # fit_region so that small regions are graded against the mesh-wide
    # noise floor, not their own tiny extent.
    mesh_bbox_diag = float(np.linalg.norm(np.ptp(full_vertices, axis=0)))

    full_face_areas = _face_areas(full_vertices, full_faces)
    total_full_area = float(full_face_areas.sum())

    # Per-face normals on the full mesh; used by cylinder fits during refit.
    try:
        full_face_normals = np.asarray(full_mesh.face_normals, dtype=np.float64)
    except Exception:
        full_face_normals = _face_normals(full_vertices, full_faces)

    regions = {}
    for r in range(n_regions):
        proxy_face_idx = np.where(proxy_labels == r)[0].astype(np.int64)
        full_face_idx = np.where(full_labels == r)[0].astype(np.int64)
        if full_face_idx.shape[0] == 0:
            continue
        area_full = float(full_face_areas[full_face_idx].sum())

        # Proxy fit. Points and normals are decoupled on purpose:
        #   - points: unique region vertices (so the plane fit is not biased
        #             by high-valence vertices repeating in the cloud)
        #   - normals: per-face normals of the region (not vertex normals;
        #              we want sharp directional info for the cylinder SVD)
        proxy_vert_idx = np.unique(proxy.faces[proxy_face_idx].flatten())
        proxy_pts = proxy.vertices[proxy_vert_idx]
        proxy_norms = proxy.face_normals[proxy_face_idx]
        proxy_fit = fit_region(proxy_pts, proxy_norms, fit_source="proxy",
                               reference_scale=mesh_bbox_diag)

        regions[r] = Region(
            id=r,
            proxy_face_indices=proxy_face_idx,
            full_face_indices=full_face_idx,
            area_full=area_full,
            area_fraction=area_full / max(total_full_area, 1e-12),
            fit=proxy_fit,
            fit_proxy=proxy_fit,
        )

    if progress_callback:
        progress_callback("intent", 88, "Full-resolution refit pass...")

    for r in regions.values():
        if r.excluded:
            continue
        verts_idx = np.unique(full_faces[r.full_face_indices].flatten())
        if verts_idx.size < 8:
            continue
        pts = full_vertices[verts_idx]
        # Use face normals for the region's faces — these match the actual
        # mesh orientation and aren't smoothed across mechanical edges the
        # way vertex normals are.
        norms = full_face_normals[r.full_face_indices]
        # Honour the manual override if the user has pinned a type on this
        # region. forced_type used to be stored-only (B2 in the initial
        # commit) — it now reaches the fitter.
        refit = fit_region(pts, norms, fit_source="fullres", forced_type=r.forced_type,
                           reference_scale=mesh_bbox_diag)
        # We always replace the fit with the full-resolution one — it has
        # the actual primitive parameters. fit_proxy is preserved as a
        # diagnostic so the frontend can show "proxy was HIGH plane,
        # fullres demoted to MEDIUM plane".
        r.fit = refit

    if progress_callback:
        progress_callback("intent", 93, "Splitting HIGH cores from MEDIUM regions...")

    # Face adjacency is shared across core splitting and every expansion
    # round — building once saves 8+ rebuilds on large meshes.
    face_adj = _build_face_adjacency(full_faces)

    # Core extraction: for MEDIUM regions, isolate the contiguous subset
    # of faces whose per-face residuals are within the HIGH band. If the
    # core is >= 45% of the region and refits as HIGH, split it off as a
    # new HIGH region. The remaining faces stay in the original region.
    full_labels, regions = _split_high_cores(
        full_labels, regions, full_vertices, full_faces,
        full_face_normals, full_face_areas, total_full_area,
        mesh_bbox_diag, min_region_faces, face_adj,
    )

    if progress_callback:
        progress_callback("intent", 96, "Expanding HIGH regions...")

    # Iterative expand + refit: expansion absorbs boundary faces from
    # MEDIUM into HIGH regions; refit may then promote MEDIUM regions
    # whose surface is now cleaner; those newly-HIGH regions can expand
    # further in the next round. Cap at 10 rounds to bound runtime.
    for _round in range(10):
        prev_high_area = sum(
            r.area_fraction for r in regions.values()
            if r.fit is not None and r.fit.confidence_class == ConfidenceClass.HIGH
        )

        full_labels, regions = _expand_high_regions(
            full_labels, regions, full_vertices, full_faces,
            full_face_normals, full_face_areas, total_full_area,
            mesh_bbox_diag, face_adj,
        )

        # Post-expansion refit: regions that lost boundary faces to
        # expansion may now have a cleaner surface that grades higher.
        for r_id in list(regions.keys()):
            r = regions[r_id]
            if r.fit is None or r.fit.confidence_class == ConfidenceClass.HIGH:
                continue
            fi = r.full_face_indices
            if len(fi) < min_region_faces:
                continue
            vert_idx = np.unique(full_faces[fi].flatten())
            if vert_idx.size < 8:
                continue
            pts = full_vertices[vert_idx]
            norms = full_face_normals[fi]
            new_fit = fit_region(pts, norms, fit_source="refit_post_expand",
                                 reference_scale=mesh_bbox_diag)
            if new_fit.confidence_class == ConfidenceClass.HIGH:
                r.fit = new_fit

        new_high_area = sum(
            r.area_fraction for r in regions.values()
            if r.fit is not None and r.fit.confidence_class == ConfidenceClass.HIGH
        )
        # Stop if this round gained less than 0.1pp.
        if new_high_area - prev_high_area < 0.001:
            break

    metrics = {
        "elapsed_sec": float(time.time() - t0),
        "target_proxy_faces": int(target_proxy),
        "min_region_faces": int(min_region_faces),
        "growth_mode": growth_mode,
    }

    state = ReconstructionState(
        mesh_id=str(session.get("mesh_id") or ""),
        proxy=proxy,
        regions=regions,
        boundaries=boundaries,
        proxy_edge_confidence=signals.confidence,
        proxy_edge_endpoints=proxy.vertices[signals.edge_vertex_pairs],
        full_face_region=full_labels.astype(np.int64),
        metrics=metrics,
    )
    session["recon_state"] = state

    if progress_callback:
        progress_callback("intent", 100, f"E0 done: {len(regions)} regions, {len(boundaries)} boundaries")

    return state.to_dict(include_regions=True)


def get_intent_state(session) -> dict:
    """Return a JSON-safe summary of the current reconstruction state."""
    state: Optional[ReconstructionState] = session.get("recon_state")
    if state is None:
        return {"available": False}
    return {
        "available": True,
        **state.to_dict(include_regions=True),
    }


def get_intent_overlays(session) -> dict:
    """Return arrays for the frontend debug overlays.

    Kept deliberately small — only the data the overlay actually needs:
      - per-full-face region id (-1 for unassigned)
      - per-region primitive type + key axis/normal so we can draw lines
      - sharp-edge segments (start/end + confidence) on the proxy mesh

    Heavy data (proxy mesh itself) is not shipped — the frontend already
    has the full-resolution mesh.
    """
    state: Optional[ReconstructionState] = session.get("recon_state")
    if state is None:
        return {"available": False}

    # Sharp edges only (cheap to draw and the only ones interesting to debug)
    confs = state.proxy_edge_confidence
    eps = state.proxy_edge_endpoints
    if confs is None or eps is None:
        sharp_edges = []
    else:
        from .boundary import SHARP_BOUNDARY_THRESHOLD
        mask = confs >= SHARP_BOUNDARY_THRESHOLD
        sharp_starts = eps[mask, 0]
        sharp_ends = eps[mask, 1]
        sharp_confs = confs[mask]
        sharp_edges = {
            "starts": sharp_starts.astype(np.float32).tolist(),
            "ends": sharp_ends.astype(np.float32).tolist(),
            "confidence": sharp_confs.astype(np.float32).tolist(),
            "n": int(mask.sum()),
        }

    # Region info needed for overlay rendering — type colour + a single
    # representative line per region (plane normal or cylinder axis).
    region_overlays = []
    for r in sorted(state.regions.values(), key=lambda r: r.id):
        fit = r.fit
        ptype = fit.type.value if fit else "unknown"
        cls = fit.confidence_class.value if fit else "low"
        gizmo = None
        if fit and fit.confidence_class in (ConfidenceClass.HIGH, ConfidenceClass.MEDIUM):
            if fit.type == PrimitiveType.PLANE:
                gizmo = {
                    "kind": "plane_normal",
                    "origin": fit.params.get("centroid", [0.0, 0.0, 0.0]),
                    "direction": fit.params["normal"],
                }
            elif fit.type == PrimitiveType.CYLINDER:
                gizmo = {
                    "kind": "cylinder_axis",
                    "origin": fit.params["center"],
                    "direction": fit.params["axis"],
                    "radius": fit.params["radius"],
                    "height": fit.params.get("height", 0.0),
                }
            elif fit.type == PrimitiveType.CONE:
                gizmo = {
                    "kind": "cone_axis",
                    "origin": fit.params["apex"],
                    "direction": fit.params["axis"],
                    "half_angle_deg": fit.params["half_angle_deg"],
                    "height": fit.params.get("height", 0.0),
                }
        region_overlays.append({
            "id": r.id,
            "type": ptype,
            "confidence_class": cls,
            "score": fit.score if fit else 0.0,
            "rmse": fit.rmse if fit else 0.0,
            "n_full_faces": int(r.full_face_indices.shape[0]),
            "area_fraction": r.area_fraction,
            "gizmo": gizmo,
        })

    return {
        "available": True,
        "n_full_faces": int(state.full_face_region.shape[0]) if state.full_face_region is not None else 0,
        "full_face_region_b64": _encode_int32(state.full_face_region) if state.full_face_region is not None else None,
        "regions": region_overlays,
        "sharp_edges": sharp_edges,
        "summary": state.summary(),
    }


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    return cross / np.maximum(norms, 1e-12)


def _build_face_adjacency(full_faces: np.ndarray) -> list:
    """Edge-sharing face adjacency. Built once per pipeline run and
    shared by core splitting and every expansion round — each rebuild
    is O(F) and we do 8+ rounds on large meshes, so sharing matters."""
    edge_to_faces = {}
    for fi in range(full_faces.shape[0]):
        tri = full_faces[fi]
        for j in range(3):
            edge = (min(int(tri[j]), int(tri[(j + 1) % 3])),
                    max(int(tri[j]), int(tri[(j + 1) % 3])))
            edge_to_faces.setdefault(edge, []).append(fi)
    face_adj = [[] for _ in range(full_faces.shape[0])]
    for edge, flist in edge_to_faces.items():
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                face_adj[flist[i]].append(flist[j])
                face_adj[flist[j]].append(flist[i])
    return face_adj


def _split_high_cores(
    full_labels: np.ndarray,
    regions: dict,
    full_vertices: np.ndarray,
    full_faces: np.ndarray,
    full_face_normals: np.ndarray,
    full_face_areas: np.ndarray,
    total_full_area: float,
    mesh_bbox_diag: float,
    min_region_faces: int,
    face_adj: list,
):
    """Extract HIGH-quality cores from MEDIUM plane regions.

    For each MEDIUM plane, compute per-face max-vertex residual against
    the fitted plane. Faces whose residual is within the HIGH band form
    the "core". If the largest connected component of core faces is big
    enough and refits as HIGH, split it off as a new region.

    This reclaims area that IS genuinely on the primitive but was stuck
    in a MEDIUM region because noisy boundary faces inflated the
    region-wide RMSE.
    """
    from .fitting import PLANE_HIGH_RMSE_REL, CYL_HIGH_RMSE_REL, CONE_HIGH_RMSE_REL
    from collections import deque

    next_id = max(regions.keys()) + 1 if regions else 0

    for r_id in list(regions.keys()):
        r = regions[r_id]
        if r.fit is None:
            continue
        if r.fit.confidence_class != ConfidenceClass.MEDIUM:
            continue
        if r.fit.type not in (PrimitiveType.PLANE, PrimitiveType.CYLINDER, PrimitiveType.CONE):
            continue

        region_bbox = float(np.linalg.norm(
            np.ptp(full_vertices[np.unique(full_faces[r.full_face_indices].flatten())], axis=0)))
        eff_bbox = max(region_bbox, mesh_bbox_diag * 0.15)

        # Per-face max-vertex residual against the primitive.
        core_faces = set()
        if r.fit.type == PrimitiveType.PLANE:
            n_vec = np.asarray(r.fit.params["normal"], dtype=np.float64)
            d_val = float(r.fit.params["d"])
            high_thr = PLANE_HIGH_RMSE_REL * eff_bbox
            for fi in r.full_face_indices:
                tri = full_vertices[full_faces[fi]]
                if np.abs(tri @ n_vec + d_val).max() <= high_thr:
                    core_faces.add(int(fi))
        elif r.fit.type == PrimitiveType.CYLINDER:
            axis = np.asarray(r.fit.params["axis"], dtype=np.float64)
            center = np.asarray(r.fit.params["center"], dtype=np.float64)
            radius = float(r.fit.params["radius"])
            high_thr = CYL_HIGH_RMSE_REL * eff_bbox
            for fi in r.full_face_indices:
                tri = full_vertices[full_faces[fi]]
                diff = tri - center
                proj = (diff @ axis)[:, None] * axis
                radial = np.linalg.norm(diff - proj, axis=1)
                if np.abs(radial - radius).max() <= high_thr:
                    core_faces.add(int(fi))
        elif r.fit.type == PrimitiveType.CONE:
            axis = np.asarray(r.fit.params["axis"], dtype=np.float64)
            apex = np.asarray(r.fit.params["apex"], dtype=np.float64)
            ha_rad = np.radians(float(r.fit.params["half_angle_deg"]))
            sin_a = float(np.sin(ha_rad))
            cos_a = float(np.cos(ha_rad))
            high_thr = CONE_HIGH_RMSE_REL * eff_bbox
            for fi in r.full_face_indices:
                tri = full_vertices[full_faces[fi]]
                diff = tri - apex
                s_ax = diff @ axis
                perp = diff - np.outer(s_ax, axis)
                r_rad = np.linalg.norm(perp, axis=1)
                if np.abs(r_rad * cos_a - np.abs(s_ax) * sin_a).max() <= high_thr:
                    core_faces.add(int(fi))

        if len(core_faces) < min_region_faces:
            continue
        # The core must be a substantial portion of the original region.
        # On a genuinely flat surface with noisy boundaries, the core is
        # 60-80% of the region. On a curved freeform surface, the "flat
        # center" is typically < 20%. The 40% gate prevents extracting
        # tiny flat patches from curved surfaces.
        if len(core_faces) < 0.45 * len(r.full_face_indices):
            continue

        # Find the largest connected component of core faces.
        visited = set()
        best_cc = []
        for start in core_faces:
            if start in visited:
                continue
            cc = []
            q = deque([start])
            visited.add(start)
            while q:
                fi = q.popleft()
                cc.append(fi)
                for nb in face_adj[fi]:
                    if nb in visited:
                        continue
                    if nb not in core_faces:
                        continue
                    visited.add(nb)
                    q.append(nb)
            if len(cc) > len(best_cc):
                best_cc = cc

        if len(best_cc) < min_region_faces:
            continue

        # Refit the core. If it grades HIGH, split it off.
        core_idx = np.asarray(best_cc, dtype=np.int64)
        vert_idx = np.unique(full_faces[core_idx].flatten())
        if vert_idx.size < 8:
            continue
        pts = full_vertices[vert_idx]
        norms = full_face_normals[core_idx]
        core_fit = fit_region(pts, norms, fit_source="core_split",
                              reference_scale=mesh_bbox_diag)
        if core_fit.confidence_class != ConfidenceClass.HIGH:
            continue

        # Split: core becomes a new HIGH region, remainder stays MEDIUM.
        remainder_idx = np.setdiff1d(r.full_face_indices, core_idx)
        core_area = float(full_face_areas[core_idx].sum())
        remainder_area = float(full_face_areas[remainder_idx].sum()) if remainder_idx.size > 0 else 0.0

        # Create new region for the core.
        for fi in core_idx:
            full_labels[fi] = next_id
        core_region = Region(
            id=next_id,
            proxy_face_indices=r.proxy_face_indices,  # approximate
            full_face_indices=core_idx,
            area_full=core_area,
            area_fraction=core_area / max(total_full_area, 1e-12),
            fit=core_fit,
            fit_proxy=r.fit_proxy,
        )
        regions[next_id] = core_region
        next_id += 1

        # Update the original region with the remainder.
        if remainder_idx.size >= min_region_faces:
            r.full_face_indices = remainder_idx
            r.area_full = remainder_area
            r.area_fraction = remainder_area / max(total_full_area, 1e-12)
            # Refit the remainder.
            rem_vert_idx = np.unique(full_faces[remainder_idx].flatten())
            if rem_vert_idx.size >= 8:
                rem_pts = full_vertices[rem_vert_idx]
                rem_norms = full_face_normals[remainder_idx]
                rem_fit = fit_region(rem_pts, rem_norms, fit_source="core_remainder",
                                     reference_scale=mesh_bbox_diag)
                r.fit = rem_fit
        else:
            # Remainder too small — absorbed faces stay labeled as orig region.
            # Mark excluded so it doesn't pollute stats.
            r.full_face_indices = remainder_idx
            r.area_full = remainder_area
            r.area_fraction = remainder_area / max(total_full_area, 1e-12)

    return full_labels, regions


def _expand_high_regions(
    full_labels: np.ndarray,
    regions: dict,
    full_vertices: np.ndarray,
    full_faces: np.ndarray,
    full_face_normals: np.ndarray,
    full_face_areas: np.ndarray,
    total_full_area: float,
    mesh_bbox_diag: float,
    face_adj: list,
):
    """BFS-expand each HIGH region into adjacent non-HIGH faces.

    For each HIGH plane/cone region, try to absorb neighbouring faces
    from MEDIUM or UNKNOWN regions whose per-vertex residual against
    the HIGH primitive is within a relaxed acceptance band. The
    expansion stops when no more adjacent faces pass the fit test.

    After expansion, the HIGH region is refitted. If it drops below
    HIGH, the expansion is rolled back.

    The per-face acceptance threshold is deliberately looser than the
    HIGH gate (uses the noisy-gate RMSE thresholds) because the
    validation refit ensures the expanded region still grades HIGH
    overall. This lets boundary faces with slightly elevated noise
    be absorbed when the surrounding core is strong enough.
    """
    from .fitting import (
        PLANE_NOISY_RMSE_REL,
        CONE_NOISY_RMSE_REL,
        CONE_HIGH_RMSE_REL,
    )
    from collections import deque

    def _commit_absorbed(r_id, r, absorbed, region_set):
        """Update labels and regions after expansion."""
        new_faces = np.asarray(sorted(region_set), dtype=np.int64)
        vert_idx = np.unique(full_faces[new_faces].flatten())
        if vert_idx.size < 8:
            return False
        pts = full_vertices[vert_idx]
        norms = full_face_normals[new_faces]
        new_fit = fit_region(pts, norms, fit_source="expand",
                             reference_scale=mesh_bbox_diag)
        if new_fit.confidence_class != ConfidenceClass.HIGH:
            return False  # roll back

        for fi in absorbed:
            old_region = int(full_labels[fi])
            full_labels[fi] = r_id
            if old_region in regions:
                src = regions[old_region]
                mask = src.full_face_indices != fi
                src.full_face_indices = src.full_face_indices[mask]
                src.area_full -= float(full_face_areas[fi])
                src.area_fraction = src.area_full / max(total_full_area, 1e-12)

        r.full_face_indices = new_faces
        r.area_full = float(full_face_areas[new_faces].sum())
        r.area_fraction = r.area_full / max(total_full_area, 1e-12)
        r.fit = new_fit
        return True

    # --- Plane expansion ---
    for r_id in list(regions.keys()):
        r = regions[r_id]
        if r.fit is None or r.fit.confidence_class != ConfidenceClass.HIGH:
            continue
        if r.fit.type != PrimitiveType.PLANE:
            continue

        n_vec = np.asarray(r.fit.params["normal"], dtype=np.float64)
        d_val = float(r.fit.params["d"])
        region_bbox = float(np.linalg.norm(
            np.ptp(full_vertices[np.unique(full_faces[r.full_face_indices].flatten())], axis=0)))
        eff_bbox = max(region_bbox, mesh_bbox_diag * 0.15)
        # Use the noisy-gate threshold for per-face acceptance — the
        # validation refit ensures overall HIGH quality after expansion.
        accept_thr = PLANE_NOISY_RMSE_REL * eff_bbox

        region_set = set(r.full_face_indices.tolist())
        absorbed = []
        q = deque()
        visited = set(region_set)
        for fi in r.full_face_indices:
            for nb in face_adj[int(fi)]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        while q:
            fi = q.popleft()
            src_region = int(full_labels[fi])
            if src_region in regions and regions[src_region].fit is not None:
                if regions[src_region].fit.confidence_class == ConfidenceClass.HIGH:
                    continue
            tri = full_vertices[full_faces[fi]]
            if np.abs(tri @ n_vec + d_val).max() > accept_thr:
                continue
            absorbed.append(fi)
            region_set.add(fi)
            for nb in face_adj[fi]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        if absorbed:
            _commit_absorbed(r_id, r, absorbed, region_set)

    # --- Cone expansion ---
    for r_id in list(regions.keys()):
        r = regions[r_id]
        if r.fit is None or r.fit.confidence_class != ConfidenceClass.HIGH:
            continue
        if r.fit.type != PrimitiveType.CONE:
            continue

        apex = np.asarray(r.fit.params["apex"], dtype=np.float64)
        axis = np.asarray(r.fit.params["axis"], dtype=np.float64)
        half_deg = float(r.fit.params["half_angle_deg"])
        half_rad = np.radians(half_deg)
        sin_a = float(np.sin(half_rad))
        cos_a = float(np.cos(half_rad))

        region_bbox = float(np.linalg.norm(
            np.ptp(full_vertices[np.unique(full_faces[r.full_face_indices].flatten())], axis=0)))
        eff_bbox = max(region_bbox, mesh_bbox_diag * 0.15)
        accept_thr = CONE_NOISY_RMSE_REL * eff_bbox

        region_set = set(r.full_face_indices.tolist())
        absorbed = []
        q = deque()
        visited = set(region_set)
        for fi in r.full_face_indices:
            for nb in face_adj[int(fi)]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        while q:
            fi = q.popleft()
            src_region = int(full_labels[fi])
            if src_region in regions and regions[src_region].fit is not None:
                if regions[src_region].fit.confidence_class == ConfidenceClass.HIGH:
                    continue
            # Cone residual: perpendicular distance to cone surface.
            tri_verts = full_vertices[full_faces[fi]]
            diff_apex = tri_verts - apex
            s_axial = diff_apex @ axis
            perp_vec = diff_apex - np.outer(s_axial, axis)
            r_radial = np.linalg.norm(perp_vec, axis=1)
            residuals = np.abs(r_radial * cos_a - np.abs(s_axial) * sin_a)
            if residuals.max() > accept_thr:
                continue
            absorbed.append(fi)
            region_set.add(fi)
            for nb in face_adj[fi]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        if absorbed:
            _commit_absorbed(r_id, r, absorbed, region_set)

    return full_labels, regions


def _encode_int32(arr: np.ndarray) -> str:
    """Base64 of an int32 little-endian buffer (compact transfer)."""
    import base64
    buf = arr.astype("<i4").tobytes()
    return base64.b64encode(buf).decode("ascii")

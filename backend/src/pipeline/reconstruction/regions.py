"""Region growing on the proxy mesh + RegionGraph construction.

Two growers are available:

  - `grow_regions` (mode="dihedral") — the original hybrid-boundary +
    soft-normal-gate grower. Works well on clean CAD meshes where real
    mechanical edges produce closed high-dihedral loops. On noisy scans
    the hard-cut edges do NOT form closed loops and the grower walks
    around them, collapsing the mesh into one mega-region.

  - `grow_regions_fit_driven` (mode="fit_driven") — picks a seed face,
    fits a plane/cylinder to its small neighborhood, then grows BFS while
    new faces stay within residual tolerance of the current primitive.
    When a face's vertices drift off the primitive (or its normal no
    longer matches the expected orientation) growth stops there. This is
    RANSAC region growing: boundaries fall out of the fit instead of
    being detected separately. It's the only thing that works on scans
    where the dihedral signal is fundamentally too noisy to close loops.

`grow_regions` (dispatcher at the top of the file) takes a `mode`
parameter so the pipeline can pick the right strategy per mesh class.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple
import numpy as np

from .boundary import BoundarySignals, SHARP_BOUNDARY_THRESHOLD
from .fitting import fit_region
from .state import Boundary, ConfidenceClass, MeshProxy, PrimitiveType


HARD_CUT_CONFIDENCE = 0.55       # never cross if hybrid confidence above this
SOFT_GATE_NORMAL_DEG = 18.0      # max angle to running region normal

# Fit-driven grower tunables. These are bbox-diagonal relative so the same
# numbers work on a 10mm boss and a 500mm bracket.
FIT_SEED_MIN_FACES = 24          # BFS from seed until >= this many faces (plane pass)
FIT_CYL_SEED_MIN_FACES = 12      # smaller seed for cylinder pass — small holes need to fit
FIT_PLANE_TOL_REL = 0.008        # max |signed distance| / bbox for plane grow
FIT_CYL_TOL_REL = 0.012          # max |radial residual| / bbox for cylinder grow
FIT_PLANE_NORMAL_DEG = 18.0      # face-normal deviation from plane normal
FIT_CYL_AXIS_PERP_DEG = 15.0     # face-normal may be this far from perpendicular-to-axis

# Cylinder-seed signature. The cheap SVD on the seed neighborhood's normals
# decides whether it's worth trying a forced cylinder fit. A cylinder surface
# has normals that lie IN a plane (sv[2] small) but actually SPAN that plane
# (sv[1] non-trivial). A flat plane has sv[1] ≈ 0 (all normals parallel) and
# gets filtered cheaply before we burn a full cylinder fit on it.
#
# These thresholds are deliberately strict. The cylinder pass can only HURT
# the segmentation by claiming territory that belongs to a plane and then
# growing across the plane via a loose tolerance. So we'd rather miss some
# cylinders than false-positive: the plane pass that runs after still
# segments the surface, just without the cylinder annotation.
CYL_SEED_SV2_MAX = 0.10          # sv[2]/sv[0] — normals planar enough
CYL_SEED_SV1_MIN = 0.20          # sv[1]/sv[0] — span the plane (not all parallel)
# Absolute radius cap relative to the WHOLE mesh bbox (not the seed
# neighborhood). A real cylindrical feature on a mechanical part is at most
# half the part's bbox; anything bigger is a noisy plane being modeled as a
# huge-radius cylinder.
CYL_SEED_MAX_RADIUS_FRAC_OF_MESH = 0.50


def grow_regions(
    proxy: MeshProxy,
    signals: BoundarySignals,
    min_region_face_count: int = 12,
    progress_callback=None,
    mode: str = "dihedral",
) -> np.ndarray:
    """Dispatch to the selected growing strategy.

    mode="dihedral" — original hybrid-boundary grower (good for clean CAD).
    mode="fit_driven" — RANSAC-style primitive-driven grower (for scans).
    """
    if mode == "fit_driven":
        return grow_regions_fit_driven(
            proxy,
            signals,
            min_region_face_count=min_region_face_count,
            progress_callback=progress_callback,
        )
    if mode != "dihedral":
        raise ValueError(f"unknown grow_regions mode: {mode!r}")
    return _grow_regions_dihedral(
        proxy,
        signals,
        min_region_face_count=min_region_face_count,
        progress_callback=progress_callback,
    )


def _grow_regions_dihedral(
    proxy: MeshProxy,
    signals: BoundarySignals,
    min_region_face_count: int = 12,
    progress_callback=None,
) -> np.ndarray:
    """Return a per-proxy-face int label vector. Labels are dense [0..K-1]."""

    n = signals.n_faces
    labels = np.full(n, -1, dtype=np.int64)
    cur_label = 0

    # Order seeds by face area so we start from the meatiest faces first;
    # the running region normal is much more stable that way.
    order = np.argsort(-proxy.face_areas)

    if progress_callback:
        progress_callback("intent.regions", 5, "Growing regions on proxy...")

    cos_gate = float(np.cos(np.radians(SOFT_GATE_NORMAL_DEG)))

    for seed in order:
        seed = int(seed)
        if labels[seed] >= 0:
            continue
        labels[seed] = cur_label
        # Running normal is area-weighted (B3 fix) — kept around so we can
        # report a stable region orientation later, but it is NO LONGER
        # used as the soft-gate reference. Comparing every new face to the
        # running mean breaks smooth curved surfaces (notably partial
        # cylinders): on a half-tube arc the mean drifts to the arc-average
        # direction and the next face looks 20°+ off the mean even though
        # it is only 3° off the previous face. The grower then carves the
        # cylinder into ~10 fragments which the plane fitter happily calls
        # HIGH plane. The fix is to gate against the LOCAL source face we
        # are growing from — sharp mechanical edges are still caught by
        # the hard-cut on boundary confidence above.
        running_normal = (
            proxy.face_normals[seed].astype(np.float64)
            * float(proxy.face_areas[seed])
        )

        q = deque([seed])
        while q:
            fi = q.popleft()
            src_normal = proxy.face_normals[fi]
            for nb, ei in signals.face_adj[fi]:
                if labels[nb] >= 0:
                    continue
                if signals.confidence[ei] > HARD_CUT_CONFIDENCE:
                    continue
                # Soft gate against the local source face — lets smooth
                # curved surfaces (cylinders) grow as one region.
                if float(np.dot(proxy.face_normals[nb], src_normal)) < cos_gate:
                    continue
                labels[nb] = cur_label
                running_normal += proxy.face_normals[nb] * float(proxy.face_areas[nb])
                q.append(nb)

        cur_label += 1

    if progress_callback:
        progress_callback("intent.regions", 65, f"Initial regions: {cur_label}")

    labels = _absorb_tiny_regions(labels, proxy, signals, min_region_face_count)

    if progress_callback:
        progress_callback("intent.regions", 100, f"Final regions: {int(labels.max()) + 1}")

    return labels


def grow_regions_fit_driven(
    proxy: MeshProxy,
    signals: BoundarySignals,
    min_region_face_count: int = 12,
    progress_callback=None,
) -> np.ndarray:
    """Primitive-driven region growing.

    For each unassigned seed, build a small BFS neighborhood, fit a plane
    or cylinder to it, then grow outward while candidate faces stay
    inside the current primitive's residual and normal-alignment
    tolerances. When the fit can no longer absorb new faces, close this
    region and pick the next seed.

    Two passes:
      1. Cylinder-seed-first scan. Walks unlabeled faces, tests each
         seed neighborhood against a cheap cylinder signature (normal
         SVD pattern: planar but spanning). Candidates that pass are
         forced through the cylinder fitter; HIGH/MEDIUM cylinder fits
         seed a region and grow. This runs first so plane growth can't
         swallow tight cylindrical features before their own seed turn.
      2. Plane-dominant pass. Area-ordered, auto-pick primitive. On
         clean CAD this is where most planes come from. On the leftover
         from pass 1, it picks up the flat faces around already-labeled
         cylinders.

    Key difference from the dihedral grower: the boundary between regions
    is whatever line the primitive fit refuses to cross. We never need
    dihedral edges to form closed loops — a noisy scan with soft CSG
    joins still segments cleanly because the residual test fires
    wherever the surface leaves the seed's primitive.
    """
    n = signals.n_faces
    labels = np.full(n, -1, dtype=np.int64)
    cur_label = 0

    bbox_diag = float(np.linalg.norm(np.ptp(proxy.vertices, axis=0)))
    if bbox_diag < 1e-9:
        bbox_diag = 1.0

    plane_tol = FIT_PLANE_TOL_REL * bbox_diag
    cyl_tol = FIT_CYL_TOL_REL * bbox_diag
    cos_plane_normal = float(np.cos(np.radians(FIT_PLANE_NORMAL_DEG)))
    sin_cyl_perp = float(np.sin(np.radians(FIT_CYL_AXIS_PERP_DEG)))

    if progress_callback:
        progress_callback("intent.regions", 5, "Fit-driven: cylinder seed pass...")

    # Pass 1: cylinder-seed-first scan.
    cur_label = _cylinder_seed_pass(
        proxy,
        signals,
        labels,
        cur_label,
        mesh_bbox_diag=bbox_diag,
        plane_tol=plane_tol,
        cyl_tol=cyl_tol,
        cos_plane_normal=cos_plane_normal,
        sin_cyl_perp=sin_cyl_perp,
    )

    if progress_callback:
        progress_callback(
            "intent.regions",
            30,
            f"Cylinder pass: {cur_label} regions; starting plane-dominant pass...",
        )

    # Pass 2: original area-ordered plane-dominant loop. Seeds: largest
    # faces first. Already-labeled cylinder faces from pass 1 are
    # skipped by the labels[seed] >= 0 check.
    order = np.argsort(-proxy.face_areas)

    for seed in order:
        seed = int(seed)
        if labels[seed] >= 0:
            continue

        # Build the seed neighborhood (small BFS ignoring already-labeled).
        seed_faces = _seed_neighborhood(seed, labels, signals.face_adj, FIT_SEED_MIN_FACES)
        if len(seed_faces) < 6:
            # Too small to fit — leave it for tiny-region absorption.
            for f in seed_faces:
                if labels[f] < 0:
                    labels[f] = cur_label
            cur_label += 1
            continue

        # Fit plane + cylinder to the seed and pick the winner honestly.
        seed_idx = np.asarray(seed_faces, dtype=np.int64)
        vert_idx = np.unique(proxy.faces[seed_idx].flatten())
        pts = proxy.vertices[vert_idx]
        norms = proxy.face_normals[seed_idx]
        fit = fit_region(pts, norms, fit_source="fit_driven_seed")

        if fit.type == PrimitiveType.UNKNOWN:
            # No primitive fits the seed — claim just the seed's faces and
            # move on. Tiny-region absorption will clean these up.
            for f in seed_faces:
                if labels[f] < 0:
                    labels[f] = cur_label
            cur_label += 1
            continue

        # Label the seed faces and seed the BFS queue with their neighbors.
        for f in seed_faces:
            if labels[f] < 0:
                labels[f] = cur_label

        q: deque = deque()
        for f in seed_faces:
            for nb, _ei in signals.face_adj[f]:
                if labels[nb] < 0:
                    q.append(nb)

        region_face_count = int((np.asarray(labels) == cur_label).sum())
        next_refit = max(2 * region_face_count, 60)

        while q:
            fi = q.popleft()
            if labels[fi] >= 0:
                continue
            if not _face_passes_fit(
                fi,
                fit,
                proxy,
                plane_tol=plane_tol,
                cyl_tol=cyl_tol,
                cos_plane_normal=cos_plane_normal,
                sin_cyl_perp=sin_cyl_perp,
            ):
                continue
            labels[fi] = cur_label
            region_face_count += 1
            for nb, _ei in signals.face_adj[fi]:
                if labels[nb] < 0:
                    q.append(nb)

            # Periodic refit. As the region grows, the primitive becomes
            # better determined; a better fit lets growth continue deeper
            # into the surface without drifting.
            if region_face_count >= next_refit:
                region_idx = np.where(labels == cur_label)[0].astype(np.int64)
                r_verts = np.unique(proxy.faces[region_idx].flatten())
                refit = fit_region(
                    proxy.vertices[r_verts],
                    proxy.face_normals[region_idx],
                    fit_source="fit_driven_refit",
                    forced_type=fit.type,
                )
                if refit.confidence_class != ConfidenceClass.REJECTED and refit.type == fit.type:
                    fit = refit
                next_refit = int(region_face_count * 2)

        cur_label += 1

    if progress_callback:
        progress_callback("intent.regions", 65, f"Initial fit-driven regions: {cur_label}")

    labels = _absorb_tiny_regions(labels, proxy, signals, min_region_face_count)

    if progress_callback:
        progress_callback("intent.regions", 100, f"Final regions: {int(labels.max()) + 1}")

    return labels


def _cylinder_seed_pass(
    proxy: MeshProxy,
    signals: BoundarySignals,
    labels: np.ndarray,
    cur_label: int,
    mesh_bbox_diag: float,
    plane_tol: float,
    cyl_tol: float,
    cos_plane_normal: float,
    sin_cyl_perp: float,
) -> int:
    """Walk unlabeled faces, seed any neighborhood that looks like a
    cylindrical patch, force a cylinder fit, and grow it if the fit
    grades HIGH or MEDIUM.

    The seed order is area-descending — same as the plane pass — but
    most plane-like seeds are filtered out cheaply by the normal-SVD
    signature before we ever call `fit_region`. What survives the
    signature filter and passes a forced cylinder fit gets a region.
    Anything that fails is left alone for the plane-dominant pass that
    runs after this one.

    Mutates `labels` in place. Returns the new cur_label.
    """
    order = np.argsort(-proxy.face_areas)

    for seed in order:
        seed = int(seed)
        if labels[seed] >= 0:
            continue

        seed_faces = _seed_neighborhood(seed, labels, signals.face_adj, FIT_CYL_SEED_MIN_FACES)
        if len(seed_faces) < 8:
            continue  # too small to identify a cylinder; plane pass will handle

        seed_idx = np.asarray(seed_faces, dtype=np.int64)
        seed_normals = proxy.face_normals[seed_idx]

        # Cheap cylinder signature: normals must lie on a great circle.
        # Center the normal cloud and SVD it. Cylinder → sv[2] near zero
        # (normals are planar) AND sv[1] not tiny (they span the plane).
        nc = seed_normals - seed_normals.mean(axis=0)
        try:
            _, sv, _ = np.linalg.svd(nc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if sv[0] < 1e-9:
            continue
        sv2_ratio = float(sv[2] / sv[0])
        sv1_ratio = float(sv[1] / sv[0])
        if sv2_ratio > CYL_SEED_SV2_MAX:
            continue  # normals not planar enough — likely a curved freeform patch
        if sv1_ratio < CYL_SEED_SV1_MIN:
            continue  # normals all parallel — this is a plane, skip

        # Passed the signature. Try a forced cylinder fit.
        vert_idx = np.unique(proxy.faces[seed_idx].flatten())
        pts = proxy.vertices[vert_idx]
        fit = fit_region(
            pts,
            seed_normals,
            fit_source="cyl_seed",
            forced_type=PrimitiveType.CYLINDER,
        )
        if fit.type != PrimitiveType.CYLINDER:
            continue
        # Accept HIGH and MEDIUM. The strict gate is the radius cap below
        # — that's what filters out noisy planes interpreted as huge-radius
        # cylinders, not the confidence class.
        if fit.confidence_class not in (ConfidenceClass.HIGH, ConfidenceClass.MEDIUM):
            continue
        # Absolute radius cap relative to the WHOLE mesh. A real mechanical
        # cylinder is at most ~half the part's bbox; a 5x-bbox-radius
        # cylinder is just a slightly-curved noisy plane and growing one
        # would eat into adjacent plane regions on the next pass.
        radius = float(fit.params.get("radius", 0.0))
        if radius > mesh_bbox_diag * CYL_SEED_MAX_RADIUS_FRAC_OF_MESH:
            continue

        # Seed the region with the entire seed neighborhood, then BFS
        # outward using the same residual/normal gate as the main grower.
        # We use a TENTATIVE label here — the grown region only commits if
        # the final refit grades HIGH. Noisy MEDIUM cylinder seeds are
        # exactly how the cylinder pass used to eat plane area on scans.
        tentative = cur_label
        for f in seed_faces:
            if labels[f] < 0:
                labels[f] = tentative

        q: deque = deque()
        for f in seed_faces:
            for nb, _ei in signals.face_adj[f]:
                if labels[nb] < 0:
                    q.append(nb)

        region_face_count = int((labels == tentative).sum())
        next_refit = max(2 * region_face_count, 60)

        while q:
            fi = q.popleft()
            if labels[fi] >= 0:
                continue
            if not _face_passes_fit(
                fi,
                fit,
                proxy,
                plane_tol=plane_tol,
                cyl_tol=cyl_tol,
                cos_plane_normal=cos_plane_normal,
                sin_cyl_perp=sin_cyl_perp,
            ):
                continue
            labels[fi] = tentative
            region_face_count += 1
            for nb, _ei in signals.face_adj[fi]:
                if labels[nb] < 0:
                    q.append(nb)

            if region_face_count >= next_refit:
                region_idx = np.where(labels == tentative)[0].astype(np.int64)
                r_verts = np.unique(proxy.faces[region_idx].flatten())
                refit = fit_region(
                    proxy.vertices[r_verts],
                    proxy.face_normals[region_idx],
                    fit_source="cyl_seed_refit",
                    forced_type=PrimitiveType.CYLINDER,
                )
                if (
                    refit.type == PrimitiveType.CYLINDER
                    and refit.confidence_class != ConfidenceClass.REJECTED
                ):
                    fit = refit
                next_refit = int(region_face_count * 2)

        # Final validation refit on the entire grown region. If the cylinder
        # didn't actually crystallize at HIGH confidence, release the labels
        # back to -1 so the plane pass can reclaim the territory. This is
        # what stops noisy MED seeds from polluting the segmentation.
        region_mask = labels == tentative
        n_region = int(region_mask.sum())
        keep = False
        if n_region >= 12:
            region_idx = np.where(region_mask)[0].astype(np.int64)
            r_verts = np.unique(proxy.faces[region_idx].flatten())
            final = fit_region(
                proxy.vertices[r_verts],
                proxy.face_normals[region_idx],
                fit_source="cyl_seed_final",
                forced_type=PrimitiveType.CYLINDER,
            )
            if (
                final.type == PrimitiveType.CYLINDER
                and final.confidence_class == ConfidenceClass.HIGH
                and float(final.params.get("radius", 0.0))
                    <= mesh_bbox_diag * CYL_SEED_MAX_RADIUS_FRAC_OF_MESH
            ):
                keep = True

        if keep:
            cur_label += 1
        else:
            labels[region_mask] = -1

    return cur_label


def _seed_neighborhood(
    seed: int,
    labels: np.ndarray,
    face_adj: List[List[Tuple[int, int]]],
    target: int,
) -> List[int]:
    """BFS from `seed`, expanding into unlabeled faces only, until we
    have at least `target` faces (or run out of reachable unlabeled ones).
    """
    out = [seed]
    visited = {seed}
    q: deque = deque([seed])
    while q and len(out) < target:
        f = q.popleft()
        for nb, _ei in face_adj[f]:
            if nb in visited:
                continue
            if labels[nb] >= 0:
                continue
            visited.add(nb)
            out.append(nb)
            if len(out) >= target:
                break
            q.append(nb)
    return out


def _face_passes_fit(
    fi: int,
    fit,
    proxy: MeshProxy,
    plane_tol: float,
    cyl_tol: float,
    cos_plane_normal: float,
    sin_cyl_perp: float,
) -> bool:
    """Return True if face `fi` can be absorbed into the current fit.

    Uses the MAX residual across the face's three vertices (not the mean)
    so we don't let a barely-off face slip in just because one vertex
    happens to land on the primitive.
    """
    if fit.type == PrimitiveType.PLANE:
        n = np.asarray(fit.params["normal"], dtype=np.float64)
        d = float(fit.params["d"])
        tri = proxy.vertices[proxy.faces[fi]]
        residuals = np.abs(tri @ n + d)
        if residuals.max() > plane_tol:
            return False
        face_normal = proxy.face_normals[fi]
        if abs(float(np.dot(face_normal, n))) < cos_plane_normal:
            return False
        return True

    if fit.type == PrimitiveType.CYLINDER:
        axis = np.asarray(fit.params["axis"], dtype=np.float64)
        center = np.asarray(fit.params["center"], dtype=np.float64)
        radius = float(fit.params["radius"])
        tri = proxy.vertices[proxy.faces[fi]]
        diff = tri - center
        proj = (diff @ axis)[:, None] * axis
        radial = np.linalg.norm(diff - proj, axis=1)
        residuals = np.abs(radial - radius)
        if residuals.max() > cyl_tol:
            return False
        # Face normal should be (approximately) perpendicular to the axis
        # on the surface of a cylinder. sin of the angle = |dot(n, axis)|.
        face_normal = proxy.face_normals[fi]
        if abs(float(np.dot(face_normal, axis))) > sin_cyl_perp:
            return False
        return True

    return False


def _absorb_tiny_regions(
    labels: np.ndarray,
    proxy: MeshProxy,
    signals: BoundarySignals,
    min_face_count: int,
) -> np.ndarray:
    """Absorb regions with < min_face_count faces into the best-matching
    larger neighbor (the one we share the longest soft boundary with).

    "Best" prefers neighbors whose mean normal is close to ours, so we don't
    glue across ridges that the grower correctly refused to cross.
    """
    n_labels = int(labels.max()) + 1 if labels.size else 0
    if n_labels <= 1:
        return labels

    # Per-region face count and mean normal
    counts = np.bincount(labels, minlength=n_labels)
    region_normal = np.zeros((n_labels, 3), dtype=np.float64)
    for fi in range(labels.shape[0]):
        region_normal[labels[fi]] += proxy.face_normals[fi]
    norms = np.linalg.norm(region_normal, axis=1, keepdims=True)
    region_normal_unit = region_normal / np.maximum(norms, 1e-12)

    changed = True
    while changed:
        changed = False
        # Build per-region adjacency on the fly so it stays consistent.
        region_neighbors: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        for ei in range(signals.n_edges):
            a = labels[signals.edge_face_pairs[ei, 0]]
            b = labels[signals.edge_face_pairs[ei, 1]]
            if a == b:
                continue
            region_neighbors[int(a)][int(b)].append(ei)
            region_neighbors[int(b)][int(a)].append(ei)

        for r in range(n_labels):
            if counts[r] == 0 or counts[r] >= min_face_count:
                continue
            neigh = region_neighbors.get(r, {})
            if not neigh:
                continue
            best_nb = None
            best_score = -1.0
            for nb, edge_ids in neigh.items():
                if counts[nb] == 0:
                    continue
                shared = len(edge_ids)
                normal_sim = float(np.dot(region_normal_unit[r], region_normal_unit[nb]))
                score = 0.6 * normal_sim + 0.4 * (shared / max(counts[r], 1))
                if score > best_score:
                    best_score = score
                    best_nb = nb
            if best_nb is None:
                continue
            mask = labels == r
            labels[mask] = best_nb
            counts[best_nb] += counts[r]
            counts[r] = 0
            region_normal[best_nb] += region_normal[r]
            n_b = np.linalg.norm(region_normal[best_nb])
            if n_b > 1e-12:
                region_normal_unit[best_nb] = region_normal[best_nb] / n_b
            changed = True

    # Contiguously relabel
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    out = np.empty_like(labels)
    for fi in range(labels.shape[0]):
        out[fi] = remap[int(labels[fi])]
    return out


def build_region_graph(
    labels: np.ndarray,
    signals: BoundarySignals,
) -> List[Boundary]:
    """Aggregate edge-level signals into RegionGraph Boundary entries."""
    pair_data: Dict[Tuple[int, int], Dict] = {}
    for ei in range(signals.n_edges):
        a = int(labels[signals.edge_face_pairs[ei, 0]])
        b = int(labels[signals.edge_face_pairs[ei, 1]])
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        d = pair_data.setdefault(key, {
            "confs": [],
            "diheds": [],
        })
        d["confs"].append(float(signals.confidence[ei]))
        d["diheds"].append(float(signals.dihedral_deg[ei]))

    boundaries: List[Boundary] = []
    for (a, b), d in pair_data.items():
        confs = np.asarray(d["confs"], dtype=np.float64)
        diheds = np.asarray(d["diheds"], dtype=np.float64)
        mean_c = float(confs.mean())
        boundaries.append(Boundary(
            region_a=a,
            region_b=b,
            proxy_edge_count=int(confs.shape[0]),
            mean_confidence=mean_c,
            max_confidence=float(confs.max()),
            mean_dihedral_deg=float(diheds.mean()),
            sharp=mean_c >= SHARP_BOUNDARY_THRESHOLD,
        ))
    boundaries.sort(key=lambda x: (x.region_a, x.region_b))
    return boundaries

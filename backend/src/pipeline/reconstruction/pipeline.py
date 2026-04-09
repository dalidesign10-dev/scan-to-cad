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

    target_proxy = int(params.get("target_proxy_faces", 30000))
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
        proxy_fit = fit_region(proxy_pts, proxy_norms, fit_source="proxy")

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
        refit = fit_region(pts, norms, fit_source="fullres", forced_type=r.forced_type)
        # We always replace the fit with the full-resolution one — it has
        # the actual primitive parameters. fit_proxy is preserved as a
        # diagnostic so the frontend can show "proxy was HIGH plane,
        # fullres demoted to MEDIUM plane".
        r.fit = refit

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


def _encode_int32(arr: np.ndarray) -> str:
    """Base64 of an int32 little-endian buffer (compact transfer)."""
    import base64
    buf = arr.astype("<i4").tobytes()
    return base64.b64encode(buf).decode("ascii")

"""Region growing on the proxy mesh + RegionGraph construction.

The grower is intentionally simple — we use boundary confidence as the
*cost of crossing*, plus a normal-similarity gate to avoid leaks across
gentle ridges that the boundary signal underrates.

  - Hard cut: never cross an edge whose confidence > HARD_CUT
  - Soft gate: even on softer edges, require neighbour normal to agree
               with the running region average within GROW_ANGLE_DEG
  - Tiny patches < min_face_area are merged into the most-similar
    neighbor (planar criterion only — we never glue a tiny patch onto
    something whose curvature class is different)

After growing we contiguously relabel and build the RegionGraph: every
pair of regions that shares one or more edges becomes a Boundary entry
holding the mean / max confidence and dihedral.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple
import numpy as np

from .boundary import BoundarySignals, SHARP_BOUNDARY_THRESHOLD
from .state import Boundary, MeshProxy


HARD_CUT_CONFIDENCE = 0.55       # never cross if hybrid confidence above this
SOFT_GATE_NORMAL_DEG = 18.0      # max angle to running region normal


def grow_regions(
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
        running_normal = proxy.face_normals[seed].astype(np.float64).copy()
        running_count = 1

        q = deque([seed])
        while q:
            fi = q.popleft()
            for nb, ei in signals.face_adj[fi]:
                if labels[nb] >= 0:
                    continue
                if signals.confidence[ei] > HARD_CUT_CONFIDENCE:
                    continue
                # Soft gate: don't cross even a mild edge if the normal jump
                # against the running region is too big.
                ref = running_normal / max(np.linalg.norm(running_normal), 1e-12)
                if float(np.dot(proxy.face_normals[nb], ref)) < cos_gate:
                    continue
                labels[nb] = cur_label
                running_normal += proxy.face_normals[nb]
                running_count += 1
                q.append(nb)

        cur_label += 1

    if progress_callback:
        progress_callback("intent.regions", 65, f"Initial regions: {cur_label}")

    labels = _absorb_tiny_regions(labels, proxy, signals, min_region_face_count)

    if progress_callback:
        progress_callback("intent.regions", 100, f"Final regions: {int(labels.max()) + 1}")

    return labels


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

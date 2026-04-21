"""Hybrid sharp-edge / boundary confidence on the proxy mesh.

The signal is the union of three cheap, well-understood primitives:

  1. dihedral_angle             — angle between adjacent face normals
  2. normal_discontinuity_jump  — local jump in normal compared to a smoothed
                                  one-ring average (catches transitions that
                                  the bare dihedral underrates because of
                                  Poisson smoothing)
  3. curvature_support          — finite-difference proxy for curvature
                                  change across the edge, derived from the
                                  dihedral itself but normalized by edge
                                  length so we get unit-free magnitudes

The three numbers are normalized to roughly [0,1] and combined with fixed
weights into a final confidence score per *edge*. We do NOT try to threshold
internally; the regions module will use the confidence as the cost of
crossing the edge during region growing, and as the per-boundary score in
the RegionGraph.

Important: this is a *signal*, not a feature detector. Wrong-confident
boundaries are worse than missing boundaries — when in doubt the score
stays low and the regions module will gladly grow across.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

from .state import MeshProxy


# Tunables. These are NOT thresholds — they only set the saturation point of
# each normalized signal so combinations behave reasonably.
DIHEDRAL_SATURATION_DEG = 60.0
NORMAL_JUMP_SATURATION = 0.6   # 1 - cos(angle); 0.6 ≈ 56°
CURVATURE_SATURATION = 8.0     # dihedral / edge_len units, scale-aware

WEIGHT_DIHEDRAL = 0.55
WEIGHT_NORMAL_JUMP = 0.30
WEIGHT_CURVATURE = 0.15

SHARP_BOUNDARY_THRESHOLD = 0.45  # used by RegionGraph to mark "sharp"


class BoundarySignals:
    """Edge-indexed boundary signals computed once per proxy.

    Stores everything as flat arrays so the region grower can index them
    in tight loops.
    """

    def __init__(
        self,
        proxy: MeshProxy,
        edge_face_pairs: np.ndarray,    # (E, 2) int64
        edge_vertex_pairs: np.ndarray,  # (E, 2) int64 — for visualisation
        face_adj: List[List[Tuple[int, int]]],  # face -> [(neighbor_face, edge_id)]
        dihedral_deg: np.ndarray,       # (E,) float
        confidence: np.ndarray,         # (E,) float in [0,1]
        edge_lengths: np.ndarray,       # (E,) float
    ):
        self.proxy = proxy
        self.edge_face_pairs = edge_face_pairs
        self.edge_vertex_pairs = edge_vertex_pairs
        self.face_adj = face_adj
        self.dihedral_deg = dihedral_deg
        self.confidence = confidence
        self.edge_lengths = edge_lengths

    @property
    def n_edges(self) -> int:
        return int(self.edge_face_pairs.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.proxy.faces.shape[0])


def compute_boundary_signals(
    proxy: MeshProxy,
    progress_callback=None,
) -> BoundarySignals:
    """Build BoundarySignals for the given proxy mesh."""

    if progress_callback:
        progress_callback("intent.boundary", 5, "Building proxy edge map...")

    edge_face_pairs, edge_vertex_pairs = _build_manifold_edge_pairs(proxy.faces)

    if progress_callback:
        progress_callback("intent.boundary", 25, "Computing dihedral angles...")

    n_edges = edge_face_pairs.shape[0]
    fa = proxy.face_normals[edge_face_pairs[:, 0]]
    fb = proxy.face_normals[edge_face_pairs[:, 1]]
    cos_d = np.clip((fa * fb).sum(axis=1), -1.0, 1.0)
    dihedral_deg = np.degrees(np.arccos(cos_d))

    edge_lengths = np.linalg.norm(
        proxy.vertices[edge_vertex_pairs[:, 0]]
        - proxy.vertices[edge_vertex_pairs[:, 1]],
        axis=1,
    )

    if progress_callback:
        progress_callback("intent.boundary", 45, "Building face adjacency...")

    face_adj: List[List[Tuple[int, int]]] = [[] for _ in range(proxy.faces.shape[0])]
    for ei in range(n_edges):
        a, b = int(edge_face_pairs[ei, 0]), int(edge_face_pairs[ei, 1])
        face_adj[a].append((b, ei))
        face_adj[b].append((a, ei))

    if progress_callback:
        progress_callback("intent.boundary", 60, "Computing smoothed normal jump...")

    smoothed_normals = _smoothed_face_normals(proxy.face_normals, face_adj)
    # Normal discontinuity jump per edge: how much the smoothed average
    # disagrees with the raw normals on each side.
    sa = smoothed_normals[edge_face_pairs[:, 0]]
    sb = smoothed_normals[edge_face_pairs[:, 1]]
    jump_a = 1.0 - np.clip((fa * sa).sum(axis=1), -1.0, 1.0)
    jump_b = 1.0 - np.clip((fb * sb).sum(axis=1), -1.0, 1.0)
    normal_jump = np.maximum(jump_a, jump_b)  # 0..2

    if progress_callback:
        progress_callback("intent.boundary", 75, "Computing curvature-derived support...")

    # Scale-aware curvature proxy: dihedral angle per unit edge length, then
    # divided by the median edge length so it ends up roughly unitless.
    median_edge = float(np.median(edge_lengths)) if edge_lengths.size else 1.0
    median_edge = max(median_edge, 1e-9)
    safe_lengths = np.maximum(edge_lengths, 1e-9)
    curvature_support = dihedral_deg / safe_lengths * median_edge

    if progress_callback:
        progress_callback("intent.boundary", 88, "Combining hybrid confidence...")

    sig_dihedral = np.clip(dihedral_deg / DIHEDRAL_SATURATION_DEG, 0.0, 1.0)
    sig_normal_jump = np.clip(normal_jump / NORMAL_JUMP_SATURATION, 0.0, 1.0)
    sig_curv = np.clip(curvature_support / CURVATURE_SATURATION, 0.0, 1.0)

    confidence = (
        WEIGHT_DIHEDRAL * sig_dihedral
        + WEIGHT_NORMAL_JUMP * sig_normal_jump
        + WEIGHT_CURVATURE * sig_curv
    )
    confidence = np.clip(confidence, 0.0, 1.0)

    if progress_callback:
        progress_callback("intent.boundary", 100, f"{n_edges:,} interior edges scored")

    return BoundarySignals(
        proxy=proxy,
        edge_face_pairs=edge_face_pairs,
        edge_vertex_pairs=edge_vertex_pairs,
        face_adj=face_adj,
        dihedral_deg=dihedral_deg,
        confidence=confidence,
        edge_lengths=edge_lengths,
    )


def _build_manifold_edge_pairs(faces: np.ndarray):
    """Return (edge_face_pairs, edge_vertex_pairs) for interior edges only.

    Boundary edges (used by exactly 1 face) are dropped — they don't have a
    dihedral signal. Non-manifold edges (3+ faces) are also dropped because
    we cannot reason about a single dihedral there.
    """
    # Build (sorted_v0, sorted_v1) -> [(face_idx, opposite_v_in_face_order)]
    edge_to_faces: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = defaultdict(list)
    for fi in range(faces.shape[0]):
        f = faces[fi]
        for i in range(3):
            a = int(f[i]); b = int(f[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_to_faces[key].append((fi, key[0], key[1]))

    pairs_face = []
    pairs_vert = []
    for key, entries in edge_to_faces.items():
        if len(entries) != 2:
            continue
        f0 = entries[0][0]
        f1 = entries[1][0]
        pairs_face.append((f0, f1))
        pairs_vert.append((key[0], key[1]))

    if not pairs_face:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 2), dtype=np.int64),
        )
    return (
        np.asarray(pairs_face, dtype=np.int64),
        np.asarray(pairs_vert, dtype=np.int64),
    )


def _smoothed_face_normals(
    face_normals: np.ndarray,
    face_adj: List[List[Tuple[int, int]]],
) -> np.ndarray:
    """One-pass area-agnostic smoothing of face normals over the 1-ring."""
    out = np.zeros_like(face_normals)
    for fi in range(face_normals.shape[0]):
        acc = face_normals[fi].copy()
        for nb, _ in face_adj[fi]:
            acc += face_normals[nb]
        n = np.linalg.norm(acc)
        if n > 1e-12:
            out[fi] = acc / n
        else:
            out[fi] = face_normals[fi]
    return out

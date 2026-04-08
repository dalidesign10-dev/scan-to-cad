"""Data model for the Phase E0 mechanical intent reconstruction layer.

Single source of truth for everything the intent pipeline produces. Held on
the FastAPI session as `SESSION["recon_state"]` so endpoints never have to
juggle scattered dicts.

Naming intent (kept deliberately small):

  MeshProxy           — decimated copy of the cleaned mesh used for
                        segmentation/boundary work + a face_map back to the
                        full-resolution mesh.
  Region              — a connected set of full-resolution face indices that
                        the system believes belong to one analytic surface.
  Boundary            — a shared edge between two regions, with a confidence
                        score from the hybrid sharp-edge detector.
  PrimitiveFit        — a plane/cylinder/unknown hypothesis attached to a
                        region, with residual stats and a confidence class.
  Constraint          — a manual override the user has placed on a region or
                        a pair of regions (force_plane, force_coaxial, ...).
                        Storage only; no enforcement logic in E0.
  ReconstructionState — the bag that holds it all. Mutating it is the only
                        legal way to update the intent layer.

These are all plain dataclasses (no ORM, no pydantic). They are converted to
plain JSON via .to_dict() when an endpoint needs to ship them to the
frontend.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class PrimitiveType(str, Enum):
    PLANE = "plane"
    CYLINDER = "cylinder"
    UNKNOWN = "unknown"


class ConfidenceClass(str, Enum):
    """Discrete buckets for primitive-fit confidence.

    The pipeline never reports a single magic number — it bins fits into
    these classes so downstream code (and the frontend) can react in a
    stable way without chasing thresholds.
    """
    HIGH = "high"        # Use as-is for downstream snapping/topology.
    MEDIUM = "medium"    # Show, but flag for review.
    LOW = "low"          # Visualise, do not trust.
    REJECTED = "rejected"  # Fit attempted, did not pass any acceptance gate.


@dataclass
class MeshProxy:
    """Decimated mesh used for intent segmentation.

    The proxy lives in tandem with the full-res cleaned mesh; `face_map` and
    `inverse_face_map` are the only thing that lets us push labels in either
    direction.
    """
    vertices: np.ndarray              # (Vp, 3) float64
    faces: np.ndarray                 # (Fp, 3) int64
    face_normals: np.ndarray          # (Fp, 3) float64
    face_areas: np.ndarray            # (Fp,)   float64
    # full_face_index -> proxy_face_index
    face_map: np.ndarray              # (F_full,) int64
    # proxy_face_index -> list of full_face_index
    inverse_face_map: List[np.ndarray]
    target_face_count: int
    full_face_count: int

    def summary(self) -> dict:
        return {
            "proxy_faces": int(self.faces.shape[0]),
            "proxy_vertices": int(self.vertices.shape[0]),
            "full_faces": int(self.full_face_count),
            "target_face_count": int(self.target_face_count),
        }


@dataclass
class PrimitiveFit:
    """A single primitive hypothesis attached to a Region.

    Always carries residual + confidence diagnostics so the frontend can
    show *why* the system trusts (or doesn't trust) a fit.
    """
    type: PrimitiveType
    params: Dict                       # type-specific (normal/d, axis/center/radius/...)
    rmse: float
    inlier_ratio: float
    residual_p95: float
    confidence_class: ConfidenceClass
    score: float                       # 0..1 monotone with confidence
    fit_source: str = "proxy"          # "proxy" | "fullres"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "params": _coerce_jsonable(self.params),
            "rmse": float(self.rmse),
            "inlier_ratio": float(self.inlier_ratio),
            "residual_p95": float(self.residual_p95),
            "confidence_class": self.confidence_class.value,
            "score": float(self.score),
            "fit_source": self.fit_source,
            "notes": self.notes,
        }


@dataclass
class Region:
    """A connected analytic-candidate region on the full-resolution mesh.

    `proxy_face_indices` lets us refit / re-grow on the proxy without
    walking the full mesh. `full_face_indices` is what every downstream
    visualization, fit, and override resolves to.
    """
    id: int
    proxy_face_indices: np.ndarray     # int64 indices into MeshProxy.faces
    full_face_indices: np.ndarray      # int64 indices into the cleaned full-res mesh
    area_full: float
    area_fraction: float               # area_full / total_full_area
    fit: Optional[PrimitiveFit] = None
    fit_proxy: Optional[PrimitiveFit] = None  # original proxy fit, kept for diagnostics
    forced_type: Optional[PrimitiveType] = None  # manual override (E0 stores only)
    excluded: bool = False             # manual "exclude bad area" flag

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "n_proxy_faces": int(self.proxy_face_indices.shape[0]),
            "n_full_faces": int(self.full_face_indices.shape[0]),
            "area_full": float(self.area_full),
            "area_fraction": float(self.area_fraction),
            "fit": self.fit.to_dict() if self.fit else None,
            "fit_proxy": self.fit_proxy.to_dict() if self.fit_proxy else None,
            "forced_type": self.forced_type.value if self.forced_type else None,
            "excluded": bool(self.excluded),
        }


@dataclass
class Boundary:
    """An adjacency entry in the RegionGraph.

    `proxy_edge_count` and `proxy_edges` capture which proxy edges are
    actually shared between the two regions. `mean_confidence` is the
    averaged hybrid sharp-edge score along the shared edges (0..1, higher
    means a more decisive boundary).
    """
    region_a: int
    region_b: int
    proxy_edge_count: int
    mean_confidence: float
    max_confidence: float
    mean_dihedral_deg: float
    sharp: bool                        # mean_confidence above the sharp threshold

    def to_dict(self) -> dict:
        return {
            "region_a": int(self.region_a),
            "region_b": int(self.region_b),
            "proxy_edge_count": int(self.proxy_edge_count),
            "mean_confidence": float(self.mean_confidence),
            "max_confidence": float(self.max_confidence),
            "mean_dihedral_deg": float(self.mean_dihedral_deg),
            "sharp": bool(self.sharp),
        }


@dataclass
class Constraint:
    """A manual user-placed constraint on the reconstruction.

    Storage-only in E0 — the pipeline acknowledges the override on
    refit, but does not yet propagate constraints into a global solver.
    """
    kind: str                          # force_plane | force_cylinder | force_coaxial | force_coplanar | merge | split | mark_sharp | exclude
    region_ids: List[int]
    payload: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "region_ids": [int(r) for r in self.region_ids],
            "payload": _coerce_jsonable(self.payload),
        }


@dataclass
class ReconstructionState:
    """Top-level intent reconstruction state.

    Held on SESSION["recon_state"]. There must be exactly one of these per
    cleaned mesh; rebuilding it is the only legal way to invalidate it.
    """
    mesh_id: str                                   # the cleaned-mesh id this state is bound to
    proxy: MeshProxy
    regions: Dict[int, Region] = field(default_factory=dict)
    boundaries: List[Boundary] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    # Optional cached arrays for the overlays endpoint:
    proxy_edge_confidence: Optional[np.ndarray] = None    # (E,) float in [0,1]
    proxy_edge_endpoints: Optional[np.ndarray] = None     # (E, 2, 3) float
    full_face_region: Optional[np.ndarray] = None         # (F_full,) int region id (-1 = unassigned)
    metrics: Dict = field(default_factory=dict)

    def summary(self) -> dict:
        n_high_plane = 0
        n_high_cyl = 0
        n_unknown = 0
        residuals_by_class: Dict[str, List[float]] = {"plane": [], "cylinder": [], "unknown": []}
        explained_area_high = 0.0
        for r in self.regions.values():
            if r.fit is None:
                n_unknown += 1
                continue
            t = r.fit.type
            if t == PrimitiveType.PLANE:
                residuals_by_class["plane"].append(r.fit.rmse)
                if r.fit.confidence_class == ConfidenceClass.HIGH:
                    n_high_plane += 1
                    explained_area_high += r.area_fraction
            elif t == PrimitiveType.CYLINDER:
                residuals_by_class["cylinder"].append(r.fit.rmse)
                if r.fit.confidence_class == ConfidenceClass.HIGH:
                    n_high_cyl += 1
                    explained_area_high += r.area_fraction
            else:
                n_unknown += 1
                residuals_by_class["unknown"].append(r.fit.rmse)

        def _mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        return {
            "n_regions": len(self.regions),
            "n_boundaries": len(self.boundaries),
            "n_high_plane_fits": int(n_high_plane),
            "n_high_cylinder_fits": int(n_high_cyl),
            "n_unknown_regions": int(n_unknown),
            "mean_rmse_plane": _mean(residuals_by_class["plane"]),
            "mean_rmse_cylinder": _mean(residuals_by_class["cylinder"]),
            "explained_area_high_pct": float(100.0 * explained_area_high),
            "proxy": self.proxy.summary(),
            **self.metrics,
        }

    def to_dict(self, include_regions: bool = True) -> dict:
        out = {
            "mesh_id": self.mesh_id,
            "summary": self.summary(),
            "constraints": [c.to_dict() for c in self.constraints],
        }
        if include_regions:
            out["regions"] = [r.to_dict() for r in sorted(self.regions.values(), key=lambda r: r.id)]
            out["boundaries"] = [b.to_dict() for b in self.boundaries]
        return out


def _coerce_jsonable(obj):
    """Recursively turn numpy arrays/scalars into plain Python lists/numbers."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(x) for x in obj]
    return obj

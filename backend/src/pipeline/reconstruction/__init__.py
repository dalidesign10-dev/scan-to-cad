"""Phase E0 — Mechanical Intent Foundation.

Isolated reconstruction layer that turns the cleaned full-resolution mesh
into a region graph with per-region primitive hypotheses (plane / cylinder
/ unknown). It does NOT replace the existing pipeline — Phase A cleanup,
Phase B cage extraction and Phase C polyhedral OCC keep working untouched.

This module is intentionally narrow:
- proxy mesh decimation + label transfer
- hybrid sharp-edge / boundary confidence
- region growing constrained by boundary confidence
- explicit RegionGraph data structure
- plane / cylinder / unknown fitting with confidence classes
- full-resolution refit pass

It deliberately does NOT yet do: trimmed-face construction, plane-plane /
plane-cylinder OCC reconstruction, fillet recovery, symmetry, dimension
snapping, B-Rep sewing.
"""

from .state import (
    ReconstructionState,
    Region,
    Boundary,
    PrimitiveFit,
    Constraint,
    MeshProxy,
    PrimitiveType,
    ConfidenceClass,
)
from .pipeline import run_intent_segmentation, get_intent_state, get_intent_overlays

__all__ = [
    "ReconstructionState",
    "Region",
    "Boundary",
    "PrimitiveFit",
    "Constraint",
    "MeshProxy",
    "PrimitiveType",
    "ConfidenceClass",
    "run_intent_segmentation",
    "get_intent_state",
    "get_intent_overlays",
]

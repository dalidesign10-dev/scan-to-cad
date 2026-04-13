"""AI-powered surface classification using Claude API.

Extracts geometric features from each region, sends them to Claude
for intelligent classification (PLANE, CYLINDER, CONE, SPHERE, FILLET,
CHAMFER, FREEFORM, UNKNOWN), and returns refined classifications with
reasoning.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional

from .state import (
    ReconstructionState, Region, PrimitiveType, ConfidenceClass,
)

logger = logging.getLogger(__name__)


def _extract_region_features(
    region: Region,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    bbox_diag: float,
    state: ReconstructionState,
) -> dict:
    """Extract geometric features for a single region."""
    fi = region.full_face_indices
    if fi is None or len(fi) == 0:
        return None

    # Get region vertices and normals
    region_faces = faces[fi]
    vert_idx = np.unique(region_faces.ravel())
    pts = vertices[vert_idx]
    norms = face_normals[fi]

    if len(pts) < 4:
        return None

    # --- Normal distribution ---
    mean_normal = norms.mean(axis=0)
    mean_normal_len = float(np.linalg.norm(mean_normal))
    if mean_normal_len > 1e-12:
        mean_normal /= mean_normal_len

    # SVD of centered normals → planarity
    centered_norms = norms - norms.mean(axis=0)
    try:
        _, sv, _ = np.linalg.svd(centered_norms, full_matrices=False)
        sv = sv / (sv[0] + 1e-12)
        normal_spread = float(sv[1])  # 0=all parallel (plane), 1=spread
        normal_flatness = float(sv[2])  # 0=planar distribution (cylinder)
    except Exception:
        normal_spread = 1.0
        normal_flatness = 1.0

    # --- Point cloud shape ---
    centered_pts = pts - pts.mean(axis=0)
    try:
        _, sv_pts, _ = np.linalg.svd(centered_pts, full_matrices=False)
        sv_pts = sv_pts / (sv_pts[0] + 1e-12)
        aspect_ratio = float(sv_pts[1])
        flatness = float(sv_pts[2])
    except Exception:
        aspect_ratio = 1.0
        flatness = 1.0

    # --- Current fit info ---
    fit_info = {}
    if region.fit:
        fit_info = {
            "current_type": region.fit.type.value,
            "rmse": round(float(region.fit.rmse), 5),
            "rmse_relative": round(float(region.fit.rmse / bbox_diag), 6),
            "inlier_ratio": round(float(region.fit.inlier_ratio), 3),
            "confidence": region.fit.confidence_class.value,
        }
        if region.fit.type == PrimitiveType.CONE:
            ha = region.fit.params.get("half_angle_deg", 0)
            fit_info["half_angle_deg"] = round(float(ha), 1)
        if region.fit.type == PrimitiveType.CYLINDER:
            fit_info["radius"] = round(float(region.fit.params.get("radius", 0)), 3)

    # --- Neighbors ---
    neighbor_types = []
    for b in state.boundaries:
        other = None
        if b.region_a == region.id:
            other = b.region_b
        elif b.region_b == region.id:
            other = b.region_a
        if other is not None:
            nr = state.regions.get(other)
            if nr and nr.fit:
                neighbor_types.append(nr.fit.type.value)

    return {
        "region_id": int(region.id),
        "n_vertices": int(len(vert_idx)),
        "n_faces": int(len(fi)),
        "area_fraction": round(float(region.area_fraction), 4),
        "normal_spread": round(normal_spread, 4),
        "normal_flatness": round(normal_flatness, 4),
        "mean_normal_coherence": round(mean_normal_len, 4),
        "point_aspect_ratio": round(aspect_ratio, 4),
        "point_flatness": round(flatness, 4),
        **fit_info,
        "n_neighbors": len(neighbor_types),
        "neighbor_types": neighbor_types[:6],
    }


def _build_classification_prompt(features_list: List[dict], part_description: str = "") -> str:
    """Build the prompt for Claude API."""
    return f"""You are an expert mechanical surface classifier for reverse-engineering 3D scans of mechanical parts.

For each mesh region below, classify its surface type based on its geometric features.

**Valid classifications:**
- PLANE — flat surface (normal_spread ≈ 0, point_flatness ≈ 0)
- CYLINDER — cylindrical surface like holes, bosses, tubes (normals on a great circle: normal_spread > 0 but normal_flatness ≈ 0)
- CONE — conical/tapered surface (normals on a small circle)
- SPHERE — spherical surface (normals spread uniformly)
- FILLET — smooth blend between two surfaces (typically small, between planes/cylinders, moderate curvature)
- CHAMFER — angled cut between surfaces (like a narrow plane between two larger planes)
- FREEFORM — organic/complex surface that doesn't fit any primitive
- UNKNOWN — too noisy or small to classify

**Key decision rules:**
- A region with normal_spread < 0.05 and point_flatness < 0.1 is almost certainly a PLANE
- A region with normal_spread > 0.1 and normal_flatness < 0.1 suggests CYLINDER
- A region with high half_angle_deg (>60°) that the fitter called "cone" is likely a FILLET or FREEFORM, not a true cone
- Small regions (area_fraction < 0.005) between two PLANE neighbors are likely FILLETS or CHAMFERS
- Low inlier_ratio (<0.6) suggests misclassification — the current_type is probably wrong
- True mechanical cones have half_angle_deg typically 15-45°; angles >60° are suspicious

{f"Part description: {part_description}" if part_description else "Part: automotive bracket (expect flat mounting surfaces, cylindrical holes, transition fillets)"}

**Region features (JSON):**
{json.dumps(features_list, indent=1)}

**Return ONLY a JSON array** (no markdown, no explanation outside the array):
[{{"region_id": <int>, "classification": "<TYPE>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}]"""


def classify_regions(
    state: ReconstructionState,
    mesh,
    api_key: Optional[str] = None,
    part_description: str = "",
) -> dict:
    """Run AI classification on all regions using Claude API.

    Args:
        state: ReconstructionState with regions and fits
        mesh: trimesh.Trimesh (preprocessed mesh)
        api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
        part_description: optional description of the part

    Returns:
        dict with classifications list and summary stats
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "No Anthropic API key. Set ANTHROPIC_API_KEY environment variable "
            "or pass api_key parameter."
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces_arr = np.asarray(mesh.faces, dtype=np.int64)
    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))

    # Compute face normals
    try:
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    except Exception:
        tri = vertices[faces_arr]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm = np.linalg.norm(cross, axis=1, keepdims=True)
        face_normals = cross / np.maximum(nrm, 1e-12)

    # Use snapped vertices if available
    if state.snap_result is not None:
        vertices = state.snap_result.snapped_vertices

    # Extract features for all regions
    features = []
    for rid, region in state.regions.items():
        feat = _extract_region_features(
            region, vertices, faces_arr, face_normals, bbox_diag, state
        )
        if feat is not None:
            features.append(feat)

    logger.info("Extracted features for %d regions", len(features))

    if not features:
        return {"classifications": [], "n_classified": 0}

    # Build prompt and call Claude API
    prompt = _build_classification_prompt(features, part_description)

    client = anthropic.Anthropic(api_key=key)

    # Batch regions into groups of 100 to avoid token limits
    BATCH_SIZE = 100
    all_classifications = []

    for batch_start in range(0, len(features), BATCH_SIZE):
        batch = features[batch_start:batch_start + BATCH_SIZE]
        batch_prompt = _build_classification_prompt(batch, part_description)

        logger.info("Calling Claude API for batch %d-%d (%d regions)...",
                     batch_start, batch_start + len(batch), len(batch))

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16384,
            messages=[{"role": "user", "content": batch_prompt}],
        )

        raw_text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:])
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

        try:
            batch_cls = json.loads(raw_text)
            all_classifications.extend(batch_cls)
        except json.JSONDecodeError as e:
            logger.error("Batch %d parse error: %s\nRaw: %s",
                         batch_start, e, raw_text[:300])
            # Try to salvage partial results
            continue

    classifications = all_classifications

    # Update state with AI classifications
    type_map = {
        "PLANE": PrimitiveType.PLANE,
        "CYLINDER": PrimitiveType.CYLINDER,
        "CONE": PrimitiveType.CONE,
    }

    n_changed = 0
    for cls in classifications:
        rid = cls.get("region_id")
        new_type = cls.get("classification", "").upper()
        region = state.regions.get(rid)
        if region is None or region.fit is None:
            continue

        old_type = region.fit.type.value.upper()
        if new_type != old_type:
            n_changed += 1

        # Store AI classification as metadata on the region
        region.ai_classification = cls

    # Summary
    type_counts = {}
    for cls in classifications:
        t = cls.get("classification", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "classifications": classifications,
        "n_classified": len(classifications),
        "n_changed": n_changed,
        "type_counts": type_counts,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
    }

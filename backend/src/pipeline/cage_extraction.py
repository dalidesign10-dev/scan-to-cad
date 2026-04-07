"""Phase B — Cage extraction.

Take the segmentation labels + per-patch fitted primitives and produce a
"cage": a small set of logical CAD faces, each backed by a single plane,
cylinder or sphere primitive. Adjacent patches whose primitive parameters
match within tolerance are merged into one cage face via union-find.

The output overwrites session["labels"] and session["patches"] with the
new cage IDs so the existing patch-selection / merge-group / plane-create
workflow keeps working unchanged on the cleaner cage faces.
"""
from collections import defaultdict
import numpy as np

from mesh_io_pkg.serialization import labels_to_transfer_file
from pipeline.segmentation import _build_face_adjacency


def _angle_between(a, b):
    """Unsigned angle in degrees between two unit vectors (handles antiparallel)."""
    d = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def _can_merge(prim_a, prim_b, tol):
    """True if two primitives are similar enough to be one logical face."""
    if prim_a.get("type") != prim_b.get("type"):
        return False

    ptype = prim_a["type"]

    if ptype == "plane":
        na = np.asarray(prim_a["normal"], dtype=float)
        nb = np.asarray(prim_b["normal"], dtype=float)
        if _angle_between(na, nb) > tol["plane_normal_deg"]:
            return False
        # Match plane offsets, accounting for possibly flipped normals
        sign = 1.0 if np.dot(na, nb) >= 0 else -1.0
        d_diff = abs(float(prim_a["d"]) - sign * float(prim_b["d"]))
        return d_diff < tol["plane_offset"]

    if ptype == "cylinder":
        aa = np.asarray(prim_a["axis"], dtype=float)
        ab = np.asarray(prim_b["axis"], dtype=float)
        if _angle_between(aa, ab) > tol["cylinder_axis_deg"]:
            return False
        ra = float(prim_a.get("radius", 0))
        rb = float(prim_b.get("radius", 0))
        if max(ra, rb) < 1e-6:
            return False
        if abs(ra - rb) / max(ra, rb) > tol["cylinder_radius_rel"]:
            return False
        # Axes must be coincident: vector between centers, projected
        # perpendicular to the axis, must be small.
        ca = np.asarray(prim_a.get("center", prim_a.get("centroid", [0, 0, 0])), dtype=float)
        cb = np.asarray(prim_b.get("center", prim_b.get("centroid", [0, 0, 0])), dtype=float)
        diff = cb - ca
        perp = diff - np.dot(diff, aa) * aa
        return float(np.linalg.norm(perp)) < max(ra, rb) * tol["cylinder_axis_offset_rel"]

    if ptype == "sphere":
        ca = np.asarray(prim_a.get("center", [0, 0, 0]), dtype=float)
        cb = np.asarray(prim_b.get("center", [0, 0, 0]), dtype=float)
        ra = float(prim_a.get("radius", 0))
        rb = float(prim_b.get("radius", 0))
        if abs(ra - rb) / max(ra, rb, 1e-6) > tol["sphere_radius_rel"]:
            return False
        return float(np.linalg.norm(ca - cb)) < tol["sphere_center"]

    return False


def extract_cage(params, progress_callback=None, session=None):
    """Build a cage of merged CAD faces from the current segmentation + fits."""
    mesh = session.get("preprocessed") or session.get("mesh")
    # Always read from the SOURCE segmentation, so re-running extract_cage
    # with different tolerances starts from the same baseline (otherwise the
    # second call would see the already-merged cage labels).
    labels = session.get("labels_source") if session.get("labels_source") is not None else session.get("labels")
    patches = session.get("patches_source") or session.get("patches")
    primitives = session.get("primitives_source") or session.get("primitives")

    if mesh is None or labels is None:
        raise ValueError("Run cleanup + segmentation first")
    if patches is None or primitives is None:
        raise ValueError("Run primitive fitting first")

    min_inlier = float(params.get("min_inlier_ratio", 0.85))
    tol = {
        "plane_normal_deg":           float(params.get("plane_normal_deg", 5.0)),
        "plane_offset":               float(params.get("plane_offset", 1.0)),
        "cylinder_axis_deg":          float(params.get("cylinder_axis_deg", 5.0)),
        "cylinder_radius_rel":        float(params.get("cylinder_radius_rel", 0.05)),
        "cylinder_axis_offset_rel":   float(params.get("cylinder_axis_offset_rel", 0.5)),
        "sphere_radius_rel":          float(params.get("sphere_radius_rel", 0.05)),
        "sphere_center":              float(params.get("sphere_center", 0.5)),
    }
    allowed_types = set(params.get("allowed_types", ("plane", "cylinder", "sphere")))

    if progress_callback:
        progress_callback("cage", 5, "Filtering qualified primitives...")

    prim_by_pid = {int(p["patch_id"]): p for p in primitives}

    qualified = {}
    for patch in patches:
        pid = int(patch["id"])
        prim = prim_by_pid.get(pid)
        if not prim:
            continue
        if prim.get("type") not in allowed_types:
            continue
        if float(prim.get("inlier_ratio", 0.0)) < min_inlier:
            continue
        qualified[pid] = prim

    if progress_callback:
        progress_callback("cage", 25, f"Building patch adjacency ({len(qualified)} qualified)...")

    n_faces = len(mesh.faces)
    face_adj, _ = _build_face_adjacency(np.asarray(mesh.faces), n_faces)

    # Patch adjacency: two patches are adjacent if they share at least one edge
    patch_adj = defaultdict(set)
    labels_np = np.asarray(labels)
    for fi in range(n_faces):
        my_label = int(labels_np[fi])
        for ni in face_adj[fi]:
            n_label = int(labels_np[ni])
            if my_label != n_label:
                patch_adj[my_label].add(n_label)
                patch_adj[n_label].add(my_label)

    if progress_callback:
        progress_callback("cage", 50, "Union-find merging compatible neighbors...")

    parent = {pid: pid for pid in qualified}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Make smaller id the root for stable cage ids
        root = min(ra, rb)
        child = max(ra, rb)
        parent[child] = root

    # Repeat until stable: merging plane A with plane B may make
    # A's params drift slightly, so we iterate until no new merges happen.
    changed = True
    rounds = 0
    while changed and rounds < 5:
        changed = False
        rounds += 1
        for pa in list(qualified.keys()):
            ra = find(pa)
            for pb in patch_adj.get(pa, ()):
                if pb not in qualified:
                    continue
                rb = find(pb)
                if ra == rb:
                    continue
                # Compare the primitives of the *root* of each component (stable choice)
                if _can_merge(qualified[ra], qualified[rb], tol):
                    union(ra, rb)
                    changed = True
                    ra = find(pa)

    if progress_callback:
        progress_callback("cage", 75, "Assigning cage IDs to faces...")

    # Build patch_id -> cage_id mapping
    cage_id_of_root = {}
    next_cage = 0
    for pid in qualified:
        root = find(pid)
        if root not in cage_id_of_root:
            cage_id_of_root[root] = next_cage
            next_cage += 1
    cage_id_of_patch = {pid: cage_id_of_root[find(pid)] for pid in qualified}

    # Special "uncovered" bucket = next_cage
    uncovered_id = next_cage
    new_labels = np.full(n_faces, uncovered_id, dtype=np.int32)
    for fi in range(n_faces):
        old = int(labels_np[fi])
        if old in cage_id_of_patch:
            new_labels[fi] = cage_id_of_patch[old]

    # Build new patches list ------------------------------------------------
    if progress_callback:
        progress_callback("cage", 85, "Building cage face metadata...")

    cage_to_root = {v: k for k, v in cage_id_of_root.items()}
    new_patches = []
    for cid in range(next_cage):
        root_pid = cage_to_root[cid]
        prim = qualified[root_pid]
        face_count = int(np.sum(new_labels == cid))
        new_patches.append({
            "id": cid,
            "face_count": face_count,
            "classification": prim.get("type", "unknown"),
            "is_fillet": False,
            "is_cage": True,
            "primitive_type": prim.get("type"),
            "source_patch_ids": [pid for pid, c in cage_id_of_patch.items() if c == cid],
        })

    n_uncovered = int(np.sum(new_labels == uncovered_id))
    if n_uncovered > 0:
        new_patches.append({
            "id": uncovered_id,
            "face_count": n_uncovered,
            "classification": "uncovered",
            "is_fillet": False,
            "is_cage": False,
        })

    # Persist over the existing keys so the rest of the pipeline reads the
    # cage as if it were a fresh segmentation.
    # Keep the original primitives + a snapshot of the original labels so the
    # user can re-run extract_cage with different tolerances. The active labels
    # are now the cage labels, but the source segmentation is preserved.
    if "primitives_source" not in session:
        session["primitives_source"] = session.get("primitives")
        session["labels_source"] = labels_np.copy()
        session["patches_source"] = session.get("patches")
    session["labels"] = new_labels
    session["patches"] = new_patches
    session["fillet_features"] = []

    # Save labels transfer file under a fresh id so the frontend re-fetches
    cage_mesh_id = str(session.get("mesh_id", "mesh")) + "_cage"
    labels_path = labels_to_transfer_file(new_labels, session["temp_dir"], cage_mesh_id)

    coverage_pct = 100.0 * (n_faces - n_uncovered) / max(n_faces, 1)

    if progress_callback:
        progress_callback("cage", 100, f"{next_cage} cage faces, {coverage_pct:.1f}% coverage")

    return {
        "labels_path": labels_path,
        "n_cage_faces": int(next_cage),
        "n_total_faces": int(n_faces),
        "n_uncovered_faces": n_uncovered,
        "coverage_pct": float(coverage_pct),
        "patches": new_patches,
        "min_inlier_ratio": min_inlier,
        "tolerances": tol,
    }

"""Surface segmentation via sharp edge detection and region growing."""
import numpy as np
from collections import deque
from mesh_io_pkg.serialization import labels_to_transfer_file


def _build_face_adjacency(faces, n_faces):
    """Build face adjacency via shared edges. Returns adj list and edge-to-face map."""
    edge_to_faces = {}
    for fi in range(n_faces):
        f = faces[fi]
        for i in range(3):
            edge = tuple(sorted([f[i], f[(i+1) % 3]]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)

    adj = [[] for _ in range(n_faces)]
    for edge, flist in edge_to_faces.items():
        for i in range(len(flist)):
            for j in range(i+1, len(flist)):
                adj[flist[i]].append(flist[j])
                adj[flist[j]].append(flist[i])
    return adj, edge_to_faces


def _compute_dihedral_angles(faces, face_normals, edge_to_faces):
    """Compute dihedral angle for each edge between two faces.
    Returns dict: (face_i, face_j) -> angle_degrees
    """
    dihedral = {}
    for edge, flist in edge_to_faces.items():
        if len(flist) != 2:
            continue
        fi, fj = flist[0], flist[1]
        n1 = face_normals[fi]
        n2 = face_normals[fj]
        dot = np.clip(np.dot(n1, n2), -1, 1)
        angle = np.degrees(np.arccos(dot))
        dihedral[(fi, fj)] = angle
        dihedral[(fj, fi)] = angle
    return dihedral


def _region_grow_with_sharp_edges(
    n_faces, face_adj, dihedral, face_normals,
    sharp_angle=25.0, grow_angle=12.0
):
    """Region growing that respects sharp edges.

    - sharp_angle: edges with dihedral angle > this are never crossed (hard boundary)
    - grow_angle: within a region, faces must have normals within this angle of the seed
    """
    labels = np.full(n_faces, -1, dtype=np.int32)
    current_label = 0

    # Sort faces by area of neighbors (start with well-connected faces)
    face_order = list(range(n_faces))

    for seed in face_order:
        if labels[seed] >= 0:
            continue

        queue = deque([seed])
        labels[seed] = current_label
        region_normals = [face_normals[seed]]
        region_normal_avg = face_normals[seed].copy()

        while queue:
            fi = queue.popleft()
            for neighbor in face_adj[fi]:
                if labels[neighbor] >= 0:
                    continue

                # Check sharp edge: never cross
                pair = (fi, neighbor)
                edge_angle = dihedral.get(pair, 0)
                if edge_angle > sharp_angle:
                    continue

                # Check normal consistency with region average
                dot_avg = np.dot(face_normals[neighbor], region_normal_avg / (np.linalg.norm(region_normal_avg) + 1e-10))
                if dot_avg < np.cos(np.radians(grow_angle)):
                    continue

                # Check normal consistency with immediate neighbor
                dot_local = np.dot(face_normals[neighbor], face_normals[fi])
                if dot_local < np.cos(np.radians(sharp_angle)):
                    continue

                labels[neighbor] = current_label
                queue.append(neighbor)

                # Update running average normal
                region_normal_avg += face_normals[neighbor]

        current_label += 1

    return labels, current_label


def _merge_small_patches(labels, face_adj, face_normals, min_faces=30):
    """Merge tiny patches into best-matching neighbor."""
    n_faces = len(labels)
    unique_labels = np.unique(labels)
    label_counts = {l: int(np.sum(labels == l)) for l in unique_labels}

    # Compute average normal per patch
    patch_normals = {}
    for l in unique_labels:
        mask = labels == l
        avg_n = face_normals[mask].mean(axis=0)
        norm = np.linalg.norm(avg_n)
        if norm > 1e-10:
            avg_n /= norm
        patch_normals[l] = avg_n

    for label in unique_labels:
        if label_counts.get(label, 0) >= min_faces:
            continue
        face_indices = np.where(labels == label)[0]
        if len(face_indices) == 0:
            continue

        # Find neighboring patches and pick the one with most similar normal
        neighbor_labels = {}
        for fi in face_indices:
            for ni in face_adj[fi]:
                nl = labels[ni]
                if nl != label:
                    neighbor_labels[nl] = neighbor_labels.get(nl, 0) + 1

        if not neighbor_labels:
            continue

        # Pick neighbor with most similar normal AND shared boundary
        my_normal = patch_normals.get(label, np.zeros(3))
        best_label = None
        best_score = -2
        for nl, count in neighbor_labels.items():
            nl_normal = patch_normals.get(nl, np.zeros(3))
            similarity = np.dot(my_normal, nl_normal)
            # Weight by boundary length (count) and normal similarity
            score = similarity * 0.7 + (count / max(len(face_indices), 1)) * 0.3
            if score > best_score:
                best_score = score
                best_label = nl

        if best_label is not None:
            labels[face_indices] = best_label

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    for i in range(n_faces):
        labels[i] = remap[labels[i]]

    return labels, len(unique)


def _aggressive_merge_coplanar(labels, face_adj, face_normals, vertices, faces,
                                merge_angle_deg=10.0, planarity_thresh_deg=18.0):
    """Iteratively merge adjacent patches that are both planar and coplanar.

    Only merges patches where:
    - Both patches have low normal variation (are individually planar)
    - Their average normals are within merge_angle
    - Their plane offsets are within distance threshold (truly coplanar, not parallel)
    """
    n_faces = len(labels)
    cos_thresh = np.cos(np.radians(merge_angle_deg))
    plan_cos = np.cos(np.radians(planarity_thresh_deg))

    # Compute model size for distance threshold
    bbox = np.ptp(vertices, axis=0)
    model_size = np.linalg.norm(bbox)
    coplanar_dist_thresh = model_size * 0.015  # 1.5% of model size (more lenient)

    # Iterate until stable
    for iteration in range(20):
        unique_labels = np.unique(labels)

        # Compute average normal, planarity, and centroid per patch
        patch_normals = {}
        patch_centroids = {}
        patch_planar = {}  # is the patch planar?
        patch_d = {}  # plane offset

        for l in unique_labels:
            mask = labels == l
            face_indices = np.where(mask)[0]
            norms = face_normals[face_indices]
            avg_n = norms.mean(axis=0)
            norm_len = np.linalg.norm(avg_n)
            if norm_len > 1e-10:
                avg_n /= norm_len
            patch_normals[l] = avg_n

            # Planarity check: 90th percentile (tolerate noise outliers)
            dots = np.clip(norms @ avg_n, -1, 1)
            p10_dot = float(np.percentile(dots, 10))
            patch_planar[l] = p10_dot > plan_cos

            # Compute centroid of patch vertices
            vert_ids = np.unique(faces[face_indices].flatten())
            centroid = vertices[vert_ids].mean(axis=0)
            patch_centroids[l] = centroid
            patch_d[l] = float(-np.dot(avg_n, centroid))

        # Find adjacent patch pairs and their boundary length
        adjacency = {}  # (l1, l2) -> shared edge count
        for fi in range(n_faces):
            li = labels[fi]
            for ni in face_adj[fi]:
                lj = labels[ni]
                if li != lj:
                    key = (min(li, lj), max(li, lj))
                    adjacency[key] = adjacency.get(key, 0) + 1

        # Build merge candidates: BOTH patches must be planar, normals similar, AND coplanar
        merges = []
        for (l1, l2), boundary in adjacency.items():
            # Both must be planar
            if not (patch_planar[l1] and patch_planar[l2]):
                continue

            n1 = patch_normals[l1]
            n2 = patch_normals[l2]
            similarity = np.dot(n1, n2)
            if similarity < cos_thresh:
                continue

            # Coplanar check: distance from one centroid to the other's plane
            c1 = patch_centroids[l1]
            c2 = patch_centroids[l2]
            n_avg = (n1 + n2)
            n_avg /= np.linalg.norm(n_avg)
            dist = abs(np.dot(c1 - c2, n_avg))
            if dist > coplanar_dist_thresh:
                continue

            merges.append((similarity, boundary, l1, l2))

        if not merges:
            break

        # Sort by score and apply merges (using union-find to avoid conflicts)
        merges.sort(reverse=True)
        parent = {l: l for l in unique_labels}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                # Smaller label absorbs larger (for stability)
                if rx < ry:
                    parent[ry] = rx
                else:
                    parent[rx] = ry
                return True
            return False

        merged_count = 0
        for _, _, l1, l2 in merges:
            if union(l1, l2):
                merged_count += 1

        # Apply union-find result
        for i in range(n_faces):
            labels[i] = find(labels[i])

        if merged_count == 0:
            break

    # Renumber contiguously
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    for i in range(n_faces):
        labels[i] = remap[labels[i]]

    return labels, len(unique)


def _detect_fillet_bands(labels, face_adj, face_normals, dihedral, max_fillet_faces=2000):
    """Identify thin bands between two large flat patches as fillet features.

    Returns labels (with fillet faces relabeled to a 'fillet' patch per band)
    and a list of fillet metadata.

    A fillet band is a small patch that:
    - Has high normal variation (curved)
    - Borders exactly 2 distinct larger patches
    - Both bordering patches are nearly planar
    """
    n_faces = len(labels)
    unique_labels = np.unique(labels)
    patch_sizes = {l: int(np.sum(labels == l)) for l in unique_labels}

    # Compute average normal and planarity for each patch
    patch_info = {}
    for l in unique_labels:
        mask = labels == l
        norms = face_normals[mask]
        avg_n = norms.mean(axis=0)
        norm = np.linalg.norm(avg_n)
        if norm > 1e-10:
            avg_n /= norm
        # Planarity: how aligned the normals are with the average
        dots = np.clip(norms @ avg_n, -1, 1)
        max_dev = np.degrees(np.arccos(dots.min())) if len(dots) > 0 else 0
        patch_info[l] = {
            "avg_normal": avg_n,
            "size": int(np.sum(mask)),
            "max_dev": float(max_dev),
            "is_planar": max_dev < 18,  # Looser planarity for fillet neighbor check
        }

    # Find candidate fillet bands
    fillet_features = []
    for label in unique_labels:
        info = patch_info[label]

        # Must be small-to-medium and curved
        if info["size"] > max_fillet_faces or info["is_planar"]:
            continue

        # Find neighboring patches
        face_indices = np.where(labels == label)[0]
        neighbor_counts = {}
        for fi in face_indices:
            for ni in face_adj[fi]:
                nl = labels[ni]
                if nl != label:
                    neighbor_counts[nl] = neighbor_counts.get(nl, 0) + 1

        # Need at least 2 planar neighbors larger than this patch (or comparable)
        large_neighbors = [(nl, c) for nl, c in neighbor_counts.items()
                           if patch_info[nl]["size"] >= info["size"] * 0.5
                           and patch_info[nl]["is_planar"]]

        if len(large_neighbors) < 2:
            continue

        # Pick the two with largest boundary
        large_neighbors.sort(key=lambda x: x[1], reverse=True)
        n1_label, n2_label = large_neighbors[0][0], large_neighbors[1][0]
        n1 = patch_info[n1_label]["avg_normal"]
        n2 = patch_info[n2_label]["avg_normal"]
        angle_between = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1, 1)))

        if angle_between < 15:
            continue  # Neighbors are too parallel — not a real edge

        fillet_features.append({
            "patch_id": int(label),
            "face_count": info["size"],
            "adjacent_patches": [int(n1_label), int(n2_label)],
            "edge_angle_deg": float(angle_between),
            "max_normal_dev_deg": info["max_dev"],
        })

    return fillet_features


def _classify_patch(face_normals_patch):
    """Classify a patch: planar, cylindrical, spherical, freeform.

    Uses angular deviation from mean normal and SVD of normal distribution.
    Thresholds tuned for noisy scan data.
    """
    if len(face_normals_patch) < 3:
        return "unknown"

    avg_normal = face_normals_patch.mean(axis=0)
    norm = np.linalg.norm(avg_normal)
    if norm > 1e-10:
        avg_normal /= norm

    # Angular deviation from mean
    dots = face_normals_patch @ avg_normal
    dots = np.clip(dots, -1, 1)
    angles = np.degrees(np.arccos(dots))
    p95_angle = np.percentile(angles, 95)
    mean_angle = np.mean(angles)

    # PLANAR: normals all point roughly the same direction
    # Use 12° for scanned data (noise creates ~5-8° variation)
    if p95_angle < 12:
        return "planar"

    # SVD of normal distribution for further classification
    centered = face_normals_patch - face_normals_patch.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return "freeform"

    # Singular value ratios
    r_min = s[2] / (s[0] + 1e-10)  # Smallest / largest
    r_mid = s[1] / (s[0] + 1e-10)  # Middle / largest

    # CYLINDRICAL: normals lie in a plane (one small SV)
    # The axis of the cylinder is perpendicular to that plane
    if r_min < 0.25 and r_mid > 0.2 and p95_angle < 50:
        return "cylindrical"

    # SPHERICAL: normals spread equally in all directions
    if r_min > 0.4 and r_mid > 0.4 and p95_angle < 60:
        return "spherical"

    # CONICAL: normals fan out unevenly
    if r_min < 0.2 and r_mid < 0.35 and p95_angle < 40:
        return "conical"

    # CURVED: moderate variation but not clearly geometric
    if p95_angle < 35:
        return "curved"

    return "freeform"


def segment_mesh(params, progress_callback=None, session=None):
    mesh = session.get("preprocessed") or session.get("mesh")
    if mesh is None:
        raise ValueError("No mesh loaded")

    sharp_angle = params.get("angle_threshold", 25.0)
    grow_angle = params.get("grow_angle", sharp_angle * 0.6)
    min_patch_faces = params.get("min_patch_faces", 30)

    vertices = mesh.vertices
    faces = mesh.faces
    n_faces = len(faces)
    face_normals = mesh.face_normals

    if face_normals is None or len(face_normals) == 0:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals /= np.maximum(norms, 1e-10)

    if progress_callback:
        progress_callback("segment", 10, "Building adjacency and edge analysis...")

    face_adj, edge_to_faces = _build_face_adjacency(faces, n_faces)

    if progress_callback:
        progress_callback("segment", 30, "Computing dihedral angles...")

    dihedral = _compute_dihedral_angles(faces, face_normals, edge_to_faces)

    if progress_callback:
        progress_callback("segment", 50, "Region growing with sharp edge detection...")

    labels, n_patches = _region_grow_with_sharp_edges(
        n_faces, face_adj, dihedral, face_normals,
        sharp_angle=sharp_angle, grow_angle=grow_angle,
    )

    if progress_callback:
        progress_callback("segment", 75, "Merging small patches...")

    labels, n_patches = _merge_small_patches(labels, face_adj, face_normals, min_patch_faces)

    if progress_callback:
        progress_callback("segment", 80, "Aggressively merging coplanar patches...")

    # Aggressive merging: combine adjacent patches with similar normals
    merge_angle = params.get("merge_angle", 14.0)
    labels, n_patches = _aggressive_merge_coplanar(
        labels, face_adj, face_normals, vertices, faces, merge_angle
    )

    if progress_callback:
        progress_callback("segment", 85, "Detecting fillet bands...")

    # Detect fillet bands between flat patches
    fillet_features = _detect_fillet_bands(labels, face_adj, face_normals, dihedral)
    session["fillet_features"] = fillet_features

    if progress_callback:
        progress_callback("segment", 88, "Classifying patches...")

    # Build set of patch IDs that are fillet bands
    fillet_patch_ids = {f["patch_id"] for f in fillet_features}

    # Compute per-patch stats
    patches = []
    for i in range(n_patches):
        mask = labels == i
        count = int(np.sum(mask))
        if count == 0:
            continue
        patch_normals = face_normals[mask]
        classification = _classify_patch(patch_normals)

        # Override classification if this is a fillet band
        if i in fillet_patch_ids:
            classification = "fillet"

        avg_normal = patch_normals.mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-10:
            avg_normal /= norm

        patches.append({
            "id": i,
            "face_count": count,
            "classification": classification,
            "avg_normal": avg_normal.tolist(),
            "is_fillet": i in fillet_patch_ids,
        })

    if progress_callback:
        progress_callback("segment", 90, "Saving labels...")

    labels_path = labels_to_transfer_file(labels, session["temp_dir"], session["mesh_id"])

    session["labels"] = labels
    session["patches"] = patches
    # Re-segmenting invalidates any cached cage source snapshot
    session.pop("labels_source", None)
    session.pop("patches_source", None)
    session.pop("primitives_source", None)

    # Count classifications
    class_counts = {}
    for p in patches:
        c = p["classification"]
        class_counts[c] = class_counts.get(c, 0) + 1

    if progress_callback:
        progress_callback("segment", 100, "Segmentation complete")

    return {
        "labels_path": labels_path,
        "n_patches": len(patches),
        "patches": patches,
        "classification_counts": class_counts,
        "fillet_features": fillet_features,
    }

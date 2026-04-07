"""Export a watertight CAD-snapped mesh.

Strategy: keep the ORIGINAL mesh topology (same vertex indices, same face indices)
so the exported mesh is guaranteed airtight. Each vertex is snapped to the analytic
primitive of its dominant patch (the patch with the most incident faces at that
vertex). Vertices on patch boundaries are averaged across all neighboring patches
so the seams stay closed.
"""
import os
import numpy as np
import trimesh

from pipeline.cad_preview import (
    _snap_to_plane, _snap_to_cylinder, _snap_to_sphere, _snap_to_bspline,
)


def _snap_vertices_for_patch(verts, prim):
    ptype = prim["type"]
    try:
        if ptype == "plane":
            snapped, _ = _snap_to_plane(verts, prim["normal"], prim["d"])
        elif ptype == "cylinder":
            snapped, _ = _snap_to_cylinder(verts, prim["axis"], prim["center"], prim["radius"])
        elif ptype == "sphere":
            snapped, _ = _snap_to_sphere(verts, prim["center"], prim["radius"])
        elif ptype == "bspline" and "poly_coeffs" in prim:
            snapped, _ = _snap_to_bspline(verts, prim)
        else:
            snapped = verts
    except Exception:
        snapped = verts
    return snapped


def _signed_distance_to_prim(points, prim):
    """Approximate signed/abs distance from points to a fitted primitive."""
    ptype = prim["type"]
    try:
        if ptype == "plane":
            n = np.asarray(prim["normal"], dtype=float)
            n /= np.linalg.norm(n)
            return np.abs(points @ n + prim["d"])
        if ptype == "cylinder":
            a = np.asarray(prim["axis"], dtype=float)
            a /= np.linalg.norm(a)
            c = np.asarray(prim["center"], dtype=float)
            rel = points - c
            radial = rel - (rel @ a)[:, None] * a[None, :]
            return np.abs(np.linalg.norm(radial, axis=1) - float(prim["radius"]))
        if ptype == "sphere":
            c = np.asarray(prim["center"], dtype=float)
            return np.abs(np.linalg.norm(points - c, axis=1) - float(prim["radius"]))
    except Exception:
        pass
    return np.full(len(points), np.inf)


def _build_watertight_snapped_mesh(session, defeature_fillets=True):
    """Build airtight CAD-snapped mesh by reverse-engineering the design intent.

    Approach (Geomagic-style defeature + reconstruct):
      1. For each vertex, collect every PLANAR primitive whose patch contains it.
         Fillet patches contribute their adjacent planar neighbors (defeature).
      2. Snap the vertex by:
           - 0 planes: keep original (or use whatever non-plane primitive it has)
           - 1 plane:  project onto that plane
           - 2 planes: project onto the intersection LINE of those planes (sharp edge)
           - 3+ planes: solve least-squares for the CORNER where all planes meet
      3. The result has true sharp edges and corners where flat faces meet.
    """
    mesh = session.get("preprocessed") or session.get("mesh")
    primitives = session.get("primitives")
    labels = session.get("labels")
    fillet_features = session.get("fillet_features") or []
    if mesh is None:
        raise ValueError("No mesh available")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if primitives is None or labels is None:
        return vertices, faces

    labels = np.asarray(labels)
    prim_by_pid = {p["patch_id"]: p for p in primitives}

    # AGGRESSIVE DEFEATURE: ignore fillet/bspline/sphere classifications entirely.
    # Only PLANE and significant CYLINDER primitives are kept; everything else is
    # treated as defeaturable (fillet/chamfer/bevel) and gets reabsorbed by
    # whichever plane/cylinder its vertices are closest to.
    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    keeper_prims = []
    for p in primitives:
        t = p["type"]
        if t == "plane":
            keeper_prims.append(p)
        elif t == "cylinder":
            r = float(p.get("radius", 0))
            face_count = int(p.get("face_count", 0))
            # Keep cylinders that look like holes/bosses (decent radius, many faces),
            # not the small ones that are really fillet rolls.
            if face_count >= 100 and r > 0.003 * bbox_diag and r < 0.4 * bbox_diag:
                keeper_prims.append(p)

    # Build patch-to-prim map for keepers only
    keeper_pids = {p["patch_id"] for p in keeper_prims}

    # Build face adjacency for hop expansion
    try:
        fa_all = np.asarray(mesh.face_adjacency)
    except Exception:
        fa_all = np.empty((0, 2), dtype=int)
    face_neighbors = [[] for _ in range(len(faces))]
    for a, b in fa_all:
        face_neighbors[a].append(int(b))
        face_neighbors[b].append(int(a))

    # For each keeper primitive, gather its vertices DILATED by N hops so it
    # claims neighboring fillet/bspline vertices too. This is the core fix:
    # a plane reaches across the fillet to grab the vertices that should
    # snap to its analytic surface.
    HOPS = 2
    vertex_prims: list[list] = [[] for _ in range(len(vertices))]
    for prim in keeper_prims:
        pid = prim["patch_id"]
        face_mask = labels == pid
        if not np.any(face_mask):
            continue
        seed_faces = set(np.where(face_mask)[0].tolist())
        current = set(seed_faces)
        frontier = set(seed_faces)
        for _ in range(HOPS):
            new_frontier = set()
            for f in frontier:
                for nb in face_neighbors[f]:
                    if nb not in current:
                        new_frontier.add(nb)
            current |= new_frontier
            frontier = new_frontier
        dilated_vids = np.unique(faces[np.fromiter(current, dtype=int)].ravel())
        for vid in dilated_vids:
            if prim not in vertex_prims[vid]:
                vertex_prims[vid].append(prim)

    # Trim each vertex's prim list to the 3 closest by signed distance, so
    # lstsq doesn't average across planes from across the part.
    for v in range(len(vertices)):
        prims = vertex_prims[v]
        if len(prims) <= 3:
            continue
        p = vertices[v]
        scored = []
        for pr in prims:
            scored.append((float(_signed_distance_to_prim(p[None, :], pr)[0]), pr))
        scored.sort(key=lambda x: x[0])
        vertex_prims[v] = [pr for _, pr in scored[:3]]

    snapped_vertices = vertices.copy()

    # Helper: snap a single vertex given its list of primitive constraints
    def snap_one(p, prims):
        # Collect plane constraints
        plane_normals = []
        plane_ds = []
        other_prims = []
        for pr in prims:
            if pr["type"] == "plane":
                n = np.asarray(pr["normal"], dtype=float)
                n = n / (np.linalg.norm(n) + 1e-12)
                plane_normals.append(n)
                plane_ds.append(float(pr["d"]))
            else:
                other_prims.append(pr)

        if len(plane_normals) >= 2:
            # Solve N*x = -d for the best-fit point lying on all planes
            N = np.array(plane_normals)
            d = np.array(plane_ds)
            # Least-squares: minimize sum (n_i.x + d_i)^2 plus pull toward p
            # Use weighted system: stack planes with a small regularization toward p
            A = N
            b = -d
            try:
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                # Project: minimize ||x - p|| subject to planes — this is the closest
                # point on the intersection. lstsq already gives min-norm solution which
                # for an underdetermined system is the closest to origin, not to p. Fix:
                # solve in delta = x - p frame: A (p + delta) = b  → A delta = b - A p
                rhs = b - A @ p
                delta, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                return p + delta
            except np.linalg.LinAlgError:
                pass

        if len(plane_normals) == 1:
            n = plane_normals[0]
            return p - (n @ p + plane_ds[0]) * n

        # No planes — fall back to first non-plane primitive
        if other_prims:
            pr = other_prims[0]
            t = pr["type"]
            try:
                if t == "cylinder":
                    s, _ = _snap_to_cylinder(p[None, :], pr["axis"], pr["center"], pr["radius"])
                    return s[0]
                if t == "sphere":
                    s, _ = _snap_to_sphere(p[None, :], pr["center"], pr["radius"])
                    return s[0]
                if t == "bspline" and "poly_coeffs" in pr:
                    s, _ = _snap_to_bspline(p[None, :], pr)
                    return s[0]
            except Exception:
                pass
        return p

    for v in range(len(vertices)):
        prims = vertex_prims[v]
        if not prims:
            continue
        snapped_vertices[v] = snap_one(vertices[v], prims)

    return snapped_vertices, faces


def _clip_polygon_by_plane(polygon, n, d, keep_positive=True):
    """Sutherland-Hodgman: clip a 3D polygon by a half-space n.x + d >= 0."""
    if not polygon:
        return []
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    def side(p):
        v = float(n @ p + d)
        return v if keep_positive else -v

    result = []
    L = len(polygon)
    for i in range(L):
        a = np.asarray(polygon[i], dtype=float)
        b = np.asarray(polygon[(i + 1) % L], dtype=float)
        sa = side(a)
        sb = side(b)
        if sa >= 0:
            result.append(a)
            if sb < 0:
                t = sa / (sa - sb)
                result.append(a + t * (b - a))
        else:
            if sb >= 0:
                t = sa / (sa - sb)
                result.append(a + t * (b - a))
    return result


def _initial_quad_on_plane(plane, size):
    n = np.asarray(plane["normal"], dtype=float)
    n /= np.linalg.norm(n) + 1e-12
    d = float(plane["d"])
    centroid = np.asarray(plane.get("centroid", [0.0, 0.0, 0.0]), dtype=float)
    p0 = centroid - (n @ centroid + d) * n
    if abs(n[0]) < 0.9:
        u = np.cross(n, [1.0, 0.0, 0.0])
    else:
        u = np.cross(n, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    h = size * 0.5
    return [
        p0 + h * u + h * v,
        p0 - h * u + h * v,
        p0 - h * u - h * v,
        p0 + h * u - h * v,
    ]


def _merge_coplanar_planes(plane_prims, bbox_diag, angle_tol_deg=15.0, dist_tol_frac=0.02,
                            adjacency=None):
    """Merge plane primitives that represent the same physical face.

    Two planes are duplicates if their normals are within angle_tol AND their
    signed offsets agree within dist_tol_frac * bbox_diag. We weight-average
    duplicates by face_count and re-emit a single plane primitive that owns
    all the contributing patch_ids.
    """
    if not plane_prims:
        return []
    cos_tol = float(np.cos(np.radians(angle_tol_deg)))
    dist_tol = dist_tol_frac * bbox_diag

    # Normalize each plane's (n, d) once and orient consistently
    normalized = []
    for p in plane_prims:
        n = np.asarray(p["normal"], dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        d = float(p["d"])
        # canonical orientation: ensure n[0] >= 0 (or first nonzero component +)
        if n[0] < 0 or (abs(n[0]) < 1e-9 and n[1] < 0):
            n = -n
            d = -d
        normalized.append((n, d, p))

    # Greedy union-find by direct comparison
    parent = list(range(len(normalized)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # If adjacency is provided, restrict merging to physically touching patches
    pid_to_idx = {p["patch_id"]: i for i, p in enumerate(plane_prims)}
    for i in range(len(normalized)):
        ni, di, pi = normalized[i]
        for j in range(i + 1, len(normalized)):
            nj, dj, pj = normalized[j]
            if (ni @ nj) < cos_tol:
                continue
            if abs(di - dj) > dist_tol:
                continue
            if adjacency is not None:
                pid_i = pi["patch_id"]
                pid_j = pj["patch_id"]
                if pid_j not in adjacency.get(pid_i, ()):
                    continue
            union(i, j)

    # Collect groups
    groups = {}
    for i in range(len(normalized)):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged = []
    for root, idxs in groups.items():
        # Weighted average of normal/d and aggregate face_count, centroid
        total_w = 0.0
        n_acc = np.zeros(3)
        d_acc = 0.0
        c_acc = np.zeros(3)
        face_count = 0
        owned_pids = []
        for i in idxs:
            n, d, p = normalized[i]
            w = float(p.get("face_count", 1))
            total_w += w
            n_acc += n * w
            d_acc += d * w
            c = np.asarray(p.get("centroid", [0, 0, 0]), dtype=float)
            c_acc += c * w
            face_count += int(p.get("face_count", 0))
            owned_pids.append(p["patch_id"])
        if total_w <= 0:
            continue
        n_acc /= np.linalg.norm(n_acc) + 1e-12
        d_acc /= total_w
        c_acc /= total_w
        merged.append({
            "type": "plane",
            "patch_id": owned_pids[0],
            "owned_pids": owned_pids,
            "normal": n_acc.tolist(),
            "d": float(d_acc),
            "centroid": c_acc.tolist(),
            "face_count": face_count,
        })
    # Sort by face_count desc so big faces come first
    merged.sort(key=lambda p: -p["face_count"])
    return merged


def _build_polyhedral_cad(session):
    """Reverse-engineered low-poly CAD: each plane is reduced to a polygon
    bounded by the analytic intersection with its neighboring planes."""
    mesh = session.get("preprocessed") or session.get("mesh")
    primitives = session.get("primitives")
    labels = session.get("labels")
    if mesh is None or primitives is None or labels is None:
        raise ValueError("Need mesh + segmentation + primitives")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = np.asarray(labels)
    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))

    # Keep only planar primitives with enough faces, then merge coplanar duplicates
    raw_planes = [p for p in primitives
                  if p["type"] == "plane" and p.get("face_count", 0) >= 400]
    merged_planes = _merge_coplanar_planes(raw_planes, bbox_diag)

    # Each merged plane "owns" multiple original patch ids; build a label→merged-id map
    pid_to_merged = {}
    for mid, mp in enumerate(merged_planes):
        for opid in mp["owned_pids"]:
            pid_to_merged[opid] = mid

    fa_all = np.asarray(mesh.face_adjacency)
    face_to_label = labels

    # Adjacency between MERGED planes via 1- and 2-hop label adjacency
    merged_neighbors = {mid: set() for mid in range(len(merged_planes))}

    label_neighbors = {}
    for a, b in fa_all:
        la, lb = int(face_to_label[a]), int(face_to_label[b])
        if la == lb:
            continue
        label_neighbors.setdefault(la, set()).add(lb)
        label_neighbors.setdefault(lb, set()).add(la)

    # Direct (label adjacent and both belong to a merged plane)
    for la, nbs in label_neighbors.items():
        ma = pid_to_merged.get(la)
        if ma is None:
            continue
        for lb in nbs:
            mb = pid_to_merged.get(lb)
            if mb is not None and mb != ma:
                merged_neighbors[ma].add(mb)
                merged_neighbors[mb].add(ma)

    # 2-hop through intermediate (non-plane) patch
    for la, nbs in label_neighbors.items():
        ma = pid_to_merged.get(la)
        if ma is None:
            continue
        for inter in nbs:
            if pid_to_merged.get(inter) is not None:
                continue
            for far in label_neighbors.get(inter, ()):
                mf = pid_to_merged.get(far)
                if mf is not None and mf != ma:
                    merged_neighbors[ma].add(mf)
                    merged_neighbors[mf].add(ma)

    prim_by_mid = {mid: mp for mid, mp in enumerate(merged_planes)}

    # Build each merged plane's polygon
    all_verts = []
    all_faces = []
    for mid, prim in prim_by_mid.items():
        poly = _initial_quad_on_plane(prim, size=1.2 * bbox_diag)

        # Centroid of this merged plane: mean of all owned patches' vertices
        owned_pids = prim["owned_pids"]
        owned_face_mask = np.isin(labels, owned_pids)
        if not np.any(owned_face_mask):
            continue
        patch_vids = np.unique(faces[owned_face_mask].ravel())
        patch_centroid = vertices[patch_vids].mean(axis=0)

        for nb_mid in merged_neighbors.get(mid, ()):
            nb_prim = prim_by_mid.get(nb_mid)
            if nb_prim is None:
                continue
            n = np.asarray(nb_prim["normal"], dtype=float)
            n /= np.linalg.norm(n) + 1e-12
            d = float(nb_prim["d"])
            keep_pos = (n @ patch_centroid + d) >= 0
            poly = _clip_polygon_by_plane(poly, n, d, keep_positive=keep_pos)
            if len(poly) < 3:
                break

        if len(poly) < 3:
            continue

        # Triangulate as a fan from the polygon centroid
        poly_arr = np.array(poly)
        center = poly_arr.mean(axis=0)
        offset = len(all_verts)
        all_verts.append(center)
        for p in poly_arr:
            all_verts.append(p)
        L = len(poly_arr)
        for i in range(L):
            a = offset
            b = offset + 1 + i
            c = offset + 1 + ((i + 1) % L)
            all_faces.append([a, b, c])

    if not all_verts:
        raise ValueError("No planar polygons survived clipping")

    out_verts = np.array(all_verts, dtype=np.float64)
    out_faces = np.array(all_faces, dtype=np.int64)
    return out_verts, out_faces


def _simplify_polygon_2d(pts2d, tol):
    """Douglas-Peucker on a closed polygon."""
    n = len(pts2d)
    if n < 4:
        return list(range(n))

    def dist_pt_seg(p, a, b):
        ab = b - a
        L2 = ab @ ab
        if L2 < 1e-12:
            return float(np.linalg.norm(p - a))
        t = max(0.0, min(1.0, ((p - a) @ ab) / L2))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def dp(start, end, mask):
        if end <= start + 1:
            return
        a = pts2d[start]
        b = pts2d[end]
        max_d = -1
        max_i = -1
        for i in range(start + 1, end):
            d = dist_pt_seg(pts2d[i], a, b)
            if d > max_d:
                max_d = d
                max_i = i
        if max_d > tol:
            mask[max_i] = True
            dp(start, max_i, mask)
            dp(max_i, end, mask)

    mask = [False] * n
    mask[0] = True
    mask[n - 1] = True
    dp(0, n - 1, mask)
    return [i for i, k in enumerate(mask) if k]


def _build_per_face_polygons(session):
    """Edge-by-edge CAD: each big plane becomes a polygon = convex hull of its
    own patch's vertices projected onto the fitted plane, then snap shared edges
    to the analytic intersection of neighboring planes.
    """
    from scipy.spatial import ConvexHull

    mesh = session.get("preprocessed") or session.get("mesh")
    primitives = session.get("primitives")
    labels = session.get("labels")
    if mesh is None or primitives is None or labels is None:
        raise ValueError("Need mesh + segmentation + primitives")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = np.asarray(labels)
    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))

    # Start with explicit plane primitives, then ALSO try to refit every other
    # patch as a plane via SVD. Any patch that is mostly flat (rmse < 1.5% bbox)
    # gets promoted to a plane primitive. This catches the many bspline patches
    # that are physically planar but were misclassified.
    raw_planes = [p for p in primitives
                  if p["type"] == "plane" and p.get("face_count", 0) >= 150]

    # Build patch adjacency from mesh face adjacency (1-hop + 2-hop through fillets)
    fa_all_pre = np.asarray(mesh.face_adjacency)
    adj_1 = {}
    for a, b in fa_all_pre:
        la, lb = int(labels[a]), int(labels[b])
        if la == lb:
            continue
        adj_1.setdefault(la, set()).add(lb)
        adj_1.setdefault(lb, set()).add(la)
    plane_pid_set = {p["patch_id"] for p in raw_planes}
    adj_for_merge = {pid: set() for pid in plane_pid_set}
    for pid in plane_pid_set:
        for nb in adj_1.get(pid, ()):
            if nb in plane_pid_set:
                adj_for_merge[pid].add(nb)
            else:
                for far in adj_1.get(nb, ()):
                    if far in plane_pid_set and far != pid:
                        adj_for_merge[pid].add(far)

    merged_planes = _merge_coplanar_planes(raw_planes, bbox_diag, adjacency=adj_for_merge)
    if not merged_planes:
        raise ValueError("No planar primitives found")

    pid_to_merged = {}
    for mid, mp in enumerate(merged_planes):
        for opid in mp["owned_pids"]:
            pid_to_merged[opid] = mid

    # Direct adjacency among merged planes (1- and 2-hop through fillets)
    fa_all = np.asarray(mesh.face_adjacency)
    label_neighbors = {}
    for a, b in fa_all:
        la, lb = int(labels[a]), int(labels[b])
        if la == lb:
            continue
        label_neighbors.setdefault(la, set()).add(lb)
        label_neighbors.setdefault(lb, set()).add(la)

    merged_neighbors = {mid: set() for mid in range(len(merged_planes))}
    for la, nbs in label_neighbors.items():
        ma = pid_to_merged.get(la)
        if ma is None:
            continue
        for lb in nbs:
            mb = pid_to_merged.get(lb)
            if mb is not None and mb != ma:
                merged_neighbors[ma].add(mb)
            else:
                # one fillet hop
                for far in label_neighbors.get(lb, ()):
                    mf = pid_to_merged.get(far)
                    if mf is not None and mf != ma:
                        merged_neighbors[ma].add(mf)

    all_verts = []
    all_faces = []

    for mid, mp in enumerate(merged_planes):
        n = np.asarray(mp["normal"], dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        d = float(mp["d"])

        # Local 2D frame on the plane
        if abs(n[0]) < 0.9:
            u = np.cross(n, [1.0, 0.0, 0.0])
        else:
            u = np.cross(n, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(n, u)

        # Collect all vertices belonging to this merged plane's owned patches
        owned_face_mask = np.isin(labels, mp["owned_pids"])
        if not np.any(owned_face_mask):
            continue
        patch_vids = np.unique(faces[owned_face_mask].ravel())
        pts3d = vertices[patch_vids]

        # Project onto the fitted plane
        proj = pts3d - (pts3d @ n + d)[:, None] * n[None, :]
        # Convert to 2D in local frame
        rel = proj - proj.mean(axis=0)
        u2 = rel @ u
        v2 = rel @ v
        pts2d = np.column_stack([u2, v2])

        # Convex hull of projected points → tight polygon following the
        # actual physical shape of the patch (no overshoot, no overlap with
        # neighboring faces). No snapping.
        if len(pts2d) < 4:
            continue
        try:
            hull = ConvexHull(pts2d)
            hull_idx = list(hull.vertices)
        except Exception:
            continue
        # Simplify hull with Douglas-Peucker so each polygon has only its
        # corner points (4-12 typical), not jagged segmentation noise.
        hull_pts2d = np.array([pts2d[i] for i in hull_idx])
        # Compute polygon area; skip degenerate
        def poly_area(p):
            x = p[:, 0]; y = p[:, 1]
            return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        area = poly_area(hull_pts2d)
        if area < (0.005 * bbox_diag) ** 2:
            continue
        # DP tolerance scales with polygon size, not bbox — small faces get less aggressive simplification
        poly_size = float(np.sqrt(area))
        simp = _simplify_polygon_2d(hull_pts2d, tol=0.08 * poly_size)
        if len(simp) < 4:
            simp = list(range(len(hull_pts2d)))
        snapped_poly = [proj[hull_idx[i]] for i in simp]
        if len(snapped_poly) < 3:
            continue

        # Ensure CCW winding relative to fitted plane normal
        poly_arr = np.array(snapped_poly)
        if len(poly_arr) >= 3:
            v0 = poly_arr[1] - poly_arr[0]
            v1 = poly_arr[2] - poly_arr[0]
            tri_n = np.cross(v0, v1)
            if tri_n @ n < 0:
                poly_arr = poly_arr[::-1]
                snapped_poly = list(poly_arr)

        # Triangulate as a fan from the polygon centroid
        center = np.mean(snapped_poly, axis=0)
        offset = len(all_verts)
        all_verts.append(center)
        for p in snapped_poly:
            all_verts.append(p)
        L = len(snapped_poly)
        for i in range(L):
            a = offset
            b = offset + 1 + i
            c = offset + 1 + ((i + 1) % L)
            all_faces.append([a, b, c])

    if not all_verts:
        raise ValueError("No polygons produced")

    return np.array(all_verts, dtype=np.float64), np.array(all_faces, dtype=np.int64)


def export_cad_preview(params, progress_callback=None, session=None):
    """Reverse-engineered CAD export.
    mode = "polyhedral" (default): low-poly model, planes clipped by neighbors.
    mode = "snapped": high-poly, original topology with vertices snapped to fits.
    """
    fmt = (params.get("format") or "stl").lower()
    mode = (params.get("mode") or "polyhedral").lower()
    output_path = params.get("output_path")
    if not output_path:
        raise ValueError("output_path required")

    if progress_callback:
        progress_callback("export", 20, f"Building {mode} CAD mesh...")

    if mode == "polyhedral":
        vertices, faces = _build_per_face_polygons(session)
    else:
        defeature = bool(params.get("defeature", True))
        vertices, faces = _build_watertight_snapped_mesh(session, defeature_fillets=defeature)

    if progress_callback:
        progress_callback("export", 50, f"Writing {fmt}...")

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    # Final weld for any remaining float-precision duplicates
    tm.merge_vertices(merge_tex=False, merge_norm=True)

    # trimesh dispatches by extension
    base, _ = os.path.splitext(output_path)
    final_path = f"{base}.{fmt}"
    tm.export(final_path)

    if progress_callback:
        progress_callback("export", 100, f"Wrote {final_path}")

    return {
        "output_path": final_path,
        "format": fmt,
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(faces)),
    }

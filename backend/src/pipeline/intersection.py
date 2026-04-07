"""Surface-surface intersection to recover sharp edges from infinite surfaces.

This is the Point2CAD-style approach:
  1. For each pair of adjacent infinite surfaces (planes, cylinders, spheres),
     compute their analytic intersection (line, circle, ellipse, or general curve).
  2. Keep only intersections that pass near the original mesh (trim to relevant portions).
  3. The resulting curves are the candidate sharp edges of the reconstructed B-Rep.
"""
import numpy as np
from scipy.spatial import cKDTree


# ---------- Plane-Plane intersection ----------

def plane_plane_intersection(n1, d1, n2, d2):
    """Intersection of two planes.

    Plane equation: n . x + d = 0
    Returns (point, direction) where direction is the line's direction vector
    or (None, None) if planes are parallel.
    """
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    direction = np.cross(n1, n2)
    denom = np.linalg.norm(direction)
    if denom < 1e-6:
        return None, None
    direction /= denom

    # Solve for a point on the line: n1.p + d1 = 0, n2.p + d2 = 0, dir.p = 0
    A = np.array([n1, n2, direction])
    b = np.array([-d1, -d2, 0.0])
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    return point, direction


# ---------- Plane-Cylinder intersection ----------

def plane_cylinder_intersection(plane_normal, plane_d, cyl_axis, cyl_center, cyl_radius,
                                 n_samples=300):
    """Intersection of a plane and a cylinder.

    Returns a list of 3D points sampling the intersection curve (generally an ellipse).
    If the plane is parallel to the cylinder axis, intersection is 0/1/2 lines.
    """
    plane_normal = np.asarray(plane_normal, dtype=float)
    plane_normal /= np.linalg.norm(plane_normal)
    cyl_axis = np.asarray(cyl_axis, dtype=float)
    cyl_axis /= np.linalg.norm(cyl_axis)
    cyl_center = np.asarray(cyl_center, dtype=float)

    # Dot between plane normal and cylinder axis
    cos_angle = np.dot(plane_normal, cyl_axis)

    if abs(cos_angle) < 1e-3:
        # Plane is parallel to cylinder axis - intersection is 0, 1 or 2 lines
        # Find distance from cylinder center to plane
        dist = np.dot(plane_normal, cyl_center) + plane_d
        if abs(dist) > cyl_radius:
            return None  # no intersection
        # Project cyl_center onto plane
        closest = cyl_center - dist * plane_normal
        # Offset perpendicular to both plane_normal and cyl_axis
        perp = np.cross(plane_normal, cyl_axis)
        perp /= np.linalg.norm(perp)
        offset = np.sqrt(max(cyl_radius**2 - dist**2, 0))
        # Two parallel lines along cyl_axis
        points = []
        for sign in (+1, -1):
            p0 = closest + sign * offset * perp
            for t in np.linspace(-cyl_radius * 3, cyl_radius * 3, n_samples // 2):
                points.append(p0 + t * cyl_axis)
        return np.array(points)

    # General case: intersection is an ellipse
    # Parameterize cylinder: p(t,s) = cyl_center + s*cyl_axis + r*(cos(t)*u + sin(t)*v)
    # where u, v are perp to cyl_axis
    if abs(cyl_axis[0]) < 0.9:
        u = np.cross(cyl_axis, [1, 0, 0])
    else:
        u = np.cross(cyl_axis, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(cyl_axis, u)

    points = []
    for t in np.linspace(0, 2 * np.pi, n_samples, endpoint=False):
        # Point on cylinder surface (s unknown, will be solved from plane eq)
        # p = cyl_center + s*cyl_axis + r*(cos(t)*u + sin(t)*v)
        circ = cyl_radius * (np.cos(t) * u + np.sin(t) * v)
        # plane_normal . (cyl_center + s*axis + circ) + d = 0
        # s = -(plane_normal.cyl_center + plane_normal.circ + d) / (plane_normal.axis)
        s_num = -(np.dot(plane_normal, cyl_center) + np.dot(plane_normal, circ) + plane_d)
        s = s_num / cos_angle
        p = cyl_center + s * cyl_axis + circ
        points.append(p)

    return np.array(points)


# ---------- Plane-Sphere intersection ----------

def plane_sphere_intersection(plane_normal, plane_d, sphere_center, sphere_radius,
                               n_samples=60):
    """Intersection of a plane and a sphere: a circle (or empty)."""
    plane_normal = np.asarray(plane_normal, dtype=float)
    plane_normal /= np.linalg.norm(plane_normal)
    sphere_center = np.asarray(sphere_center, dtype=float)

    # Distance from sphere center to plane
    dist = np.dot(plane_normal, sphere_center) + plane_d
    if abs(dist) >= sphere_radius:
        return None

    # Projected center is the circle center
    circle_center = sphere_center - dist * plane_normal
    circle_radius = np.sqrt(sphere_radius ** 2 - dist ** 2)

    # Build orthonormal basis in the plane
    if abs(plane_normal[0]) < 0.9:
        u = np.cross(plane_normal, [1, 0, 0])
    else:
        u = np.cross(plane_normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)

    points = []
    for t in np.linspace(0, 2 * np.pi, n_samples, endpoint=False):
        p = circle_center + circle_radius * (np.cos(t) * u + np.sin(t) * v)
        points.append(p)
    return np.array(points)


# ---------- Cylinder-Cylinder intersection (sampled) ----------

def cylinder_cylinder_intersection(ax1, c1, r1, ax2, c2, r2, n_samples=200):
    """Approximate cylinder-cylinder intersection by sampling surface 1 and
    keeping points near surface 2. Good enough for visualization."""
    ax1 = np.asarray(ax1) / np.linalg.norm(ax1)
    ax2 = np.asarray(ax2) / np.linalg.norm(ax2)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)

    # Sample points on cylinder 1
    if abs(ax1[0]) < 0.9:
        u1 = np.cross(ax1, [1, 0, 0])
    else:
        u1 = np.cross(ax1, [0, 1, 0])
    u1 /= np.linalg.norm(u1)
    v1 = np.cross(ax1, u1)

    # Length to sample along axis 1
    extent = max(r1, r2) * 4
    samples_t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    samples_s = np.linspace(-extent, extent, 20)

    points = []
    tol = max(r1, r2) * 0.05
    for s in samples_s:
        for t in samples_t:
            p = c1 + s * ax1 + r1 * (np.cos(t) * u1 + np.sin(t) * v1)
            # Distance from p to cylinder 2 surface
            diff = p - c2
            proj = np.dot(diff, ax2)
            radial = diff - proj * ax2
            dist_to_axis2 = np.linalg.norm(radial)
            if abs(dist_to_axis2 - r2) < tol:
                points.append(p)

    return np.array(points) if points else None


# ---------- Adjacency: which surfaces should we intersect? ----------

def _surface_touches_patch(surf_a, surf_b, mesh_tree, mesh_vertices, trim_dist):
    """Quick check: do the two surfaces have mesh vertices within trim_dist of each other?"""
    # For now we compute all pairs; adjacency filtering can be added later
    return True


# ---------- Main intersection entry point ----------

def _compute_trim_threshold(mesh_vertices):
    bbox = np.ptp(mesh_vertices, axis=0)
    return float(np.linalg.norm(bbox)) * 0.02  # 2% of model size


def _trim_to_patches(points, tree_a, tree_b, trim_dist):
    """Keep only curve points close to BOTH patch_a and patch_b vertices.
    This ensures edges lie on the actual shared boundary, not anywhere on the mesh.
    Also splits into contiguous runs so jumps become separate polylines."""
    if points is None or len(points) == 0:
        return []
    da, _ = tree_a.query(points, k=1)
    db, _ = tree_b.query(points, k=1)
    mask = (da < trim_dist) & (db < trim_dist)
    if not np.any(mask):
        return []
    # Split into contiguous runs
    runs = []
    current = []
    for i, keep in enumerate(mask):
        if keep:
            current.append(points[i])
        else:
            if len(current) >= 2:
                runs.append(np.array(current))
            current = []
    if len(current) >= 2:
        runs.append(np.array(current))
    return runs


def intersect_surfaces(params, progress_callback=None, session=None):
    """Compute pairwise intersections of infinite surfaces and trim near the mesh.

    Returns a list of edge curves (each a polyline of 3D points).
    """
    mesh = session.get("preprocessed") or session.get("mesh")
    surfaces = session.get("infinite_surfaces")
    labels = session.get("labels")
    if mesh is None or not surfaces:
        raise ValueError("Must run detect_features before intersection")
    if labels is None:
        raise ValueError("Must run segmentation before intersection")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    labels = np.asarray(labels)
    trim_dist = _compute_trim_threshold(vertices)

    # Build per-patch vertex KDTrees, DILATED by 2-hop face neighbors so that
    # fillets/small-patch bands between two primary patches are included.
    patch_trees = {}
    unique_pids = set(int(s["patch_id"]) for s in surfaces)
    try:
        fa_all = np.asarray(mesh.face_adjacency)
    except Exception:
        fa_all = np.empty((0, 2), dtype=int)

    # Build face->face adjacency list once
    face_neighbors = [[] for _ in range(len(faces))]
    for a, b in fa_all:
        face_neighbors[a].append(b)
        face_neighbors[b].append(a)

    for pid in unique_pids:
        face_mask = labels == pid
        if not np.any(face_mask):
            continue
        # BFS 2 hops from this patch
        current = set(np.where(face_mask)[0].tolist())
        frontier = set(current)
        for _ in range(2):
            new_frontier = set()
            for f in frontier:
                for nb in face_neighbors[f]:
                    if nb not in current:
                        new_frontier.add(nb)
            current |= new_frontier
            frontier = new_frontier
        dilated_faces = np.fromiter(current, dtype=int)
        vids = np.unique(faces[dilated_faces].ravel())
        if len(vids) < 3:
            continue
        patch_trees[pid] = cKDTree(vertices[vids])

    # Build patch adjacency (1-hop + 2-hop) from mesh face adjacency.
    # 2-hop captures pairs separated by a fillet/small intermediate patch.
    patch_adjacency = set()
    try:
        fa = np.asarray(mesh.face_adjacency)  # (M, 2)
        la = labels[fa[:, 0]]
        lb = labels[fa[:, 1]]
        diff = la != lb
        neighbor_map = {}  # pid -> set of directly-adjacent pids
        for a, b in zip(la[diff], lb[diff]):
            a, b = int(a), int(b)
            neighbor_map.setdefault(a, set()).add(b)
            neighbor_map.setdefault(b, set()).add(a)
            patch_adjacency.add((min(a, b), max(a, b)))
        # 2-hop
        for p, ns in neighbor_map.items():
            for n in ns:
                for nn in neighbor_map.get(n, ()):
                    if nn != p:
                        patch_adjacency.add((min(p, nn), max(p, nn)))
    except Exception:
        patch_adjacency = None  # fallback: no adjacency filter

    if progress_callback:
        progress_callback("intersect", 10, f"Intersecting {len(surfaces)} surfaces...")

    edges = []
    n_surfs = len(surfaces)
    pair_count = 0
    total_pairs = n_surfs * (n_surfs - 1) // 2

    for i in range(n_surfs):
        for j in range(i + 1, n_surfs):
            pair_count += 1
            if progress_callback and pair_count % max(1, total_pairs // 20) == 0:
                pct = 10 + int(80 * pair_count / max(total_pairs, 1))
                progress_callback("intersect", pct,
                                  f"Pair {pair_count}/{total_pairs}")

            surf_a = surfaces[i]
            surf_b = surfaces[j]
            ta, tb = surf_a["type"], surf_b["type"]

            curve_points = None
            try:
                if ta == "infinite_plane" and tb == "infinite_plane":
                    point, direction = plane_plane_intersection(
                        surf_a["normal"], surf_a["d"],
                        surf_b["normal"], surf_b["d"],
                    )
                    if point is not None:
                        # Sample line over a large range, trim later
                        model_extent = trim_dist * 50
                        ts = np.linspace(-model_extent, model_extent, 600)
                        curve_points = point + ts[:, None] * direction

                elif ta == "infinite_plane" and tb == "infinite_cylinder":
                    curve_points = plane_cylinder_intersection(
                        surf_a["normal"], surf_a["d"],
                        surf_b["axis"], surf_b["center"], surf_b["radius"],
                    )
                elif ta == "infinite_cylinder" and tb == "infinite_plane":
                    curve_points = plane_cylinder_intersection(
                        surf_b["normal"], surf_b["d"],
                        surf_a["axis"], surf_a["center"], surf_a["radius"],
                    )

                elif ta == "infinite_plane" and tb == "infinite_sphere":
                    curve_points = plane_sphere_intersection(
                        surf_a["normal"], surf_a["d"],
                        surf_b["center"], surf_b["radius"],
                    )
                elif ta == "infinite_sphere" and tb == "infinite_plane":
                    curve_points = plane_sphere_intersection(
                        surf_b["normal"], surf_b["d"],
                        surf_a["center"], surf_a["radius"],
                    )

                elif ta == "infinite_cylinder" and tb == "infinite_cylinder":
                    curve_points = cylinder_cylinder_intersection(
                        surf_a["axis"], surf_a["center"], surf_a["radius"],
                        surf_b["axis"], surf_b["center"], surf_b["radius"],
                    )

                # Skip sphere-sphere and sphere-cylinder for now (rare and complex)
            except Exception:
                curve_points = None

            if curve_points is None or len(curve_points) < 2:
                continue

            pid_a = surf_a["patch_id"]
            pid_b = surf_b["patch_id"]
            tree_a = patch_trees.get(pid_a)
            tree_b = patch_trees.get(pid_b)
            if tree_a is None or tree_b is None:
                continue

            # Trim to segments near BOTH patches simultaneously
            runs = _trim_to_patches(curve_points, tree_a, tree_b, trim_dist)
            for run in runs:
                if len(run) < 2:
                    continue
                edges.append({
                    "patch_a": pid_a,
                    "patch_b": pid_b,
                    "type_a": ta,
                    "type_b": tb,
                    "points": run.tolist(),
                    "n_points": len(run),
                })

    session["edges"] = edges

    if progress_callback:
        progress_callback("intersect", 100, f"Found {len(edges)} edge curves")

    return {
        "edges": edges,
        "n_edges": len(edges),
    }

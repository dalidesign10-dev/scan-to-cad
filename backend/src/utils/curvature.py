"""Discrete curvature estimation on triangle meshes."""
import numpy as np
from scipy.sparse import csr_matrix


def compute_vertex_normals(vertices, faces):
    """Compute area-weighted vertex normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    face_normals /= norms

    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vertex_normals /= norms
    return vertex_normals


def compute_principal_curvatures(vertices, faces, normals, k_ring=2):
    """Estimate principal curvatures via local quadric fitting.

    For each vertex, fit a quadric surface z = ax^2 + bxy + cy^2
    in the local tangent frame, then extract principal curvatures
    from the second fundamental form.

    Returns (k1, k2) arrays where k1 >= k2.
    """
    n_verts = len(vertices)
    k1 = np.zeros(n_verts)
    k2 = np.zeros(n_verts)

    # Build adjacency
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edges.add((min(a, b), max(a, b)))

    adj = [[] for _ in range(n_verts)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # k-ring neighborhood
    def get_k_ring(v, k):
        visited = {v}
        frontier = {v}
        for _ in range(k):
            next_frontier = set()
            for u in frontier:
                for w in adj[u]:
                    if w not in visited:
                        visited.add(w)
                        next_frontier.add(w)
            frontier = next_frontier
        visited.discard(v)
        return list(visited)

    for vi in range(n_verts):
        neighbors = get_k_ring(vi, k_ring)
        if len(neighbors) < 5:
            continue

        n = normals[vi]
        p = vertices[vi]

        # Build local tangent frame
        # Pick an arbitrary tangent vector
        if abs(n[0]) < 0.9:
            t1 = np.cross(n, [1, 0, 0])
        else:
            t1 = np.cross(n, [0, 1, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        # Project neighbors into local frame
        pts = vertices[neighbors] - p
        u = pts @ t1
        v = pts @ t2
        w = pts @ n

        # Fit z = ax^2 + bxy + cy^2 (least squares)
        A = np.column_stack([u**2, u*v, v**2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
        except np.linalg.LinAlgError:
            continue

        a, b, c = coeffs

        # Principal curvatures from shape operator
        # H = a + c, K = 4ac - b^2
        H = a + c
        discriminant = max((a - c)**2 + b**2, 0)
        sqrt_disc = np.sqrt(discriminant)
        k1[vi] = H + sqrt_disc
        k2[vi] = H - sqrt_disc

    return k1, k2


def compute_face_curvatures(vertices, faces, k1_vert, k2_vert):
    """Average vertex curvatures to faces."""
    k1_face = np.mean(k1_vert[faces], axis=1)
    k2_face = np.mean(k2_vert[faces], axis=1)
    return k1_face, k2_face

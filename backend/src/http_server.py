"""HTTP API server for the scan-to-CAD pipeline. Works with browser frontend."""
import os
import sys
import uuid
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from mesh_io_pkg.mesh_io import load_mesh_file
from mesh_io_pkg.serialization import mesh_to_transfer_file, labels_to_transfer_file

app = FastAPI(title="Geomagic Claude Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session state
SESSION = {
    "mesh": None,
    "mesh_id": None,
    "preprocessed": None,
    "labels": None,
    "patches": None,
    "primitives": None,
    "features": None,
    "temp_dir": tempfile.mkdtemp(prefix="geomagic_"),
    "brep_shape": None,
    # User-defined merge groups: maps patch_id -> group_id (representative patch_id).
    # Patches sharing a group_id are treated as the same logical face downstream.
    "merge_parent": {},
    # User-created infinite planes (each: {normal, d, centroid, source_patch_ids})
    "user_planes": [],
    # Phase E0 — mechanical intent reconstruction state. Single object,
    # NOT a bag of session keys. See pipeline.reconstruction.state.
    "recon_state": None,
}


def _merge_find(pid: int) -> int:
    """Union-find: find root of patch id."""
    parent = SESSION["merge_parent"]
    if pid not in parent:
        return pid
    root = pid
    while parent.get(root, root) != root:
        root = parent[root]
    # Path compression
    cur = pid
    while parent.get(cur, cur) != root:
        nxt = parent[cur]
        parent[cur] = root
        cur = nxt
    return root


def _merge_union(a: int, b: int) -> int:
    ra, rb = _merge_find(a), _merge_find(b)
    if ra == rb:
        return ra
    root, child = (ra, rb) if ra < rb else (rb, ra)
    SESSION["merge_parent"][child] = root
    return root


def _merge_map() -> dict:
    """Return current patch_id -> group_id (root) map for all known patches."""
    out = {}
    for pid in list(SESSION["merge_parent"].keys()):
        out[pid] = _merge_find(pid)
    return out


def progress_noop(stage, pct, message=""):
    """Progress callback (no-op for HTTP, could use SSE later)."""
    print(f"  [{stage}] {pct}% - {message}")


@app.get("/api/ping")
def ping():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/load_mesh")
async def api_load_mesh(file: UploadFile = File(None), path: str = None):
    """Load mesh from uploaded file or local path."""
    try:
        if file and file.filename:
            # Save uploaded file to temp
            ext = os.path.splitext(file.filename)[1] or ".stl"
            tmp_path = os.path.join(SESSION["temp_dir"], f"upload{ext}")
            content = await file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)
            load_path = tmp_path
        elif path:
            load_path = path
        else:
            raise HTTPException(400, "Provide a file upload or path parameter")

        from pipeline.preprocessing import load_mesh
        result = load_mesh(
            {"path": load_path},
            progress_callback=progress_noop,
            session=SESSION,
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/export_brep")
def api_export_brep(params: dict = {}):
    """Export the current Phase C polyhedral B-Rep through OCC.
    Body: { format: "stl"|"step"|"brep" }
    Returns the resulting filename so the frontend can download it.
    """
    try:
        from pipeline.features import export_brep
        result = export_brep(params, progress_callback=progress_noop, session=SESSION)
        result["filename"] = os.path.basename(result["output_path"])
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/build_polyhedral_brep")
def api_build_polyhedral_brep(params: dict = {}):
    """Phase C — Decimate the cleaned mesh and lift it to a polyhedral OCC solid.
    Body: { target_faces: int (default 1500) }
    Returns the tessellated solid as {vertices, faces} for the viewport.
    """
    try:
        from pipeline.features import build_polyhedral_brep
        return build_polyhedral_brep(params, progress_callback=progress_noop, session=SESSION)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/fillet_sharp_edges")
def api_fillet_sharp_edges(params: dict = {}):
    """Phase C — Round all sharp edges of the current B-Rep.
    Body: { radius: float, min_dihedral_deg: float (default 20) }
    """
    try:
        from pipeline.features import fillet_sharp_edges
        return fillet_sharp_edges(params, progress_callback=progress_noop, session=SESSION)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/chamfer_sharp_edges")
def api_chamfer_sharp_edges(params: dict = {}):
    """Phase C — Bevel all sharp edges of the current B-Rep.
    Body: { distance: float, min_dihedral_deg: float (default 20) }
    """
    try:
        from pipeline.features import chamfer_sharp_edges
        return chamfer_sharp_edges(params, progress_callback=progress_noop, session=SESSION)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/extract_cage")
def api_extract_cage(params: dict = {}):
    """Phase B — Build a cage from segmentation + primitive fits.

    Body (all optional):
      min_inlier_ratio       float default 0.85
      plane_normal_deg       float default 5.0
      plane_offset           float default 1.0
      cylinder_axis_deg      float default 5.0
      cylinder_radius_rel    float default 0.05
      sphere_radius_rel      float default 0.05
    """
    try:
        from pipeline.cage_extraction import extract_cage
        result = extract_cage(params, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/cleanup_mesh")
def api_cleanup_mesh(params: dict = {}):
    """Phase A — Poisson cleanup of the raw scan.

    Body (all optional):
      poisson_depth      int   default 10
      sample_count       int   default 400_000
      density_cutoff     float default 0.02
      taubin_iters       int   default 5
      keep_largest_only  bool  default True
    """
    try:
        from pipeline.cleanup import cleanup_mesh
        result = cleanup_mesh(params, progress_callback=progress_noop, session=SESSION)
        # Cleanup invalidates any prior intent reconstruction.
        SESSION["recon_state"] = None
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────
# Phase E0 — Mechanical intent reconstruction
# ─────────────────────────────────────────────────────────────────────────
# These endpoints build and inspect a single ReconstructionState held on
# SESSION["recon_state"]. They do not interact with Phase A/B/C state at
# all — Rough Export and the polyhedral path are unaffected.

@app.post("/api/intent/run")
def api_intent_run(params: dict = {}):
    """Build a fresh ReconstructionState from the current cleaned mesh.

    Body (all optional):
      target_proxy_faces  int   default 30000
      min_region_faces    int   default 12
      growth_mode         str   "dihedral" (default) | "fit_driven"
    """
    try:
        from pipeline.reconstruction import run_intent_segmentation
        return run_intent_segmentation(params, progress_callback=progress_noop, session=SESSION)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/intent/state")
def api_intent_state():
    """Return the current ReconstructionState as JSON (regions + boundaries)."""
    try:
        from pipeline.reconstruction import get_intent_state
        return get_intent_state(SESSION)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/intent/overlays")
def api_intent_overlays():
    """Return debug overlay data: per-face region ids, per-region gizmos,
    and the sharp-edge segments. Used by the IntentOverlay frontend layer."""
    try:
        from pipeline.reconstruction import get_intent_overlays
        return get_intent_overlays(SESSION)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/intent/override")
def api_intent_override(params: dict = {}):
    """Record a manual override on the current ReconstructionState, and for
    force_plane / force_cylinder, immediately refit the affected regions so
    the user sees the result without rerunning the whole intent pass.

    Body: { kind: "force_plane"|"force_cylinder"|"merge"|"split"|
                  "mark_sharp"|"exclude"|"force_coaxial"|"force_coplanar",
            region_ids: [int, ...],
            payload: {...} }

    Global constraints (force_coaxial / force_coplanar) are still recorded
    only — they need a multi-region solver which is the next milestone.
    """
    try:
        import numpy as np
        from pipeline.reconstruction.state import Constraint, PrimitiveType
        from pipeline.reconstruction.fitting import fit_region
        state = SESSION.get("recon_state")
        if state is None:
            raise HTTPException(400, "No reconstruction state — run /api/intent/run first")
        kind = params.get("kind")
        if not kind:
            raise HTTPException(400, "kind required")
        region_ids = [int(r) for r in (params.get("region_ids") or [])]
        payload = params.get("payload") or {}
        c = Constraint(kind=kind, region_ids=region_ids, payload=payload)
        state.constraints.append(c)

        full_mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if full_mesh is not None:
            full_vertices = np.asarray(full_mesh.vertices, dtype=np.float64)
            full_faces = np.asarray(full_mesh.faces, dtype=np.int64)
            try:
                full_face_normals = np.asarray(full_mesh.face_normals, dtype=np.float64)
            except Exception:
                tri = full_vertices[full_faces]
                cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                nrm = np.linalg.norm(cross, axis=1, keepdims=True)
                full_face_normals = cross / np.maximum(nrm, 1e-12)
        else:
            full_vertices = full_faces = full_face_normals = None

        refit_ids = []
        for rid in region_ids:
            r = state.regions.get(rid)
            if r is None:
                continue
            if kind == "force_plane":
                r.forced_type = PrimitiveType.PLANE
                refit_ids.append(rid)
            elif kind == "force_cylinder":
                r.forced_type = PrimitiveType.CYLINDER
                refit_ids.append(rid)
            elif kind == "exclude":
                r.excluded = True

        # Immediate refit so the override endpoint does its one job. Falls
        # back silently if we don't have a full mesh on the session.
        if full_vertices is not None:
            for rid in refit_ids:
                r = state.regions[rid]
                if r.full_face_indices.size == 0:
                    continue
                verts_idx = np.unique(full_faces[r.full_face_indices].flatten())
                if verts_idx.size < 8:
                    continue
                pts = full_vertices[verts_idx]
                norms = full_face_normals[r.full_face_indices]
                r.fit = fit_region(pts, norms, fit_source="override", forced_type=r.forced_type)

        return {
            "ok": True,
            "n_constraints": len(state.constraints),
            "refit_region_ids": refit_ids,
            "summary": state.summary(),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/preprocess")
def api_preprocess(params: dict = {}):
    try:
        from pipeline.preprocessing import preprocess_mesh
        result = preprocess_mesh(params, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/merge_patches")
def api_merge_patches(params: dict = {}):
    """Merge a set of patch IDs into one logical group.
    Body: { "patch_ids": [int, ...] }
    Returns: { merge_map: {patch_id: group_id}, groups: [[ids...], ...] }
    """
    try:
        ids = params.get("patch_ids") or []
        ids = [int(i) for i in ids]
        if len(ids) >= 2:
            base = ids[0]
            for other in ids[1:]:
                _merge_union(base, other)
        # Ensure all listed ids appear in the map (even if alone)
        for i in ids:
            if i not in SESSION["merge_parent"]:
                SESSION["merge_parent"][i] = i
        mm = _merge_map()
        # Also include groups view
        groups = {}
        for pid, gid in mm.items():
            groups.setdefault(gid, []).append(pid)
        return {"merge_map": mm, "groups": list(groups.values())}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/clear_merges")
def api_clear_merges():
    SESSION["merge_parent"] = {}
    return {"merge_map": {}, "groups": []}


@app.get("/api/merge_groups")
def api_get_merges():
    mm = _merge_map()
    groups = {}
    for pid, gid in mm.items():
        groups.setdefault(gid, []).append(pid)
    return {"merge_map": mm, "groups": list(groups.values())}


@app.post("/api/infinite_plane_from_patches")
def api_infinite_plane_from_patches(params: dict = {}):
    """Fit an infinite plane to the faces belonging to the given patch IDs.
    Body: { "patch_ids": [int, ...], "extent_scale": float (default 0.25) }
    Returns an InfiniteSurface dict ready to render.
    """
    try:
        import numpy as np
        from pipeline.feature_detection import _build_infinite_plane, _compute_mesh_bbox_size

        ids = [int(i) for i in (params.get("patch_ids") or [])]
        if not ids:
            raise HTTPException(400, "patch_ids required")

        mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        labels = SESSION.get("labels")
        if mesh is None or labels is None:
            raise HTTPException(400, "Must segment mesh first")

        # Collect faces for the requested patches
        mask = np.isin(labels, np.array(ids, dtype=labels.dtype))
        face_idx = np.where(mask)[0]
        if len(face_idx) < 3:
            raise HTTPException(400, f"Not enough faces in selected patches ({len(face_idx)})")

        # Area-weighted plane fit using FACE normals — robust for noisy/curved
        # scanned patches where vertex-PCA gets biased by dense or boundary points.
        verts = np.asarray(mesh.vertices, dtype=float)
        tri = verts[np.asarray(mesh.faces[face_idx])]  # (M, 3, 3)
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        face_centroids = (v0 + v1 + v2) / 3.0
        # Use trimesh's stable normals when available, otherwise from cross
        try:
            face_normals = np.asarray(mesh.face_normals[face_idx], dtype=float)
        except Exception:
            face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

        total_area = float(areas.sum())
        if total_area < 1e-12:
            raise HTTPException(400, "Selected patch has zero area")

        # Weighted average normal (area-weighted)
        normal = (face_normals * areas[:, None]).sum(axis=0)
        normal /= (np.linalg.norm(normal) + 1e-12)
        # Weighted centroid (area-weighted)
        centroid = (face_centroids * areas[:, None]).sum(axis=0) / total_area
        d = -float(np.dot(normal, centroid))

        # Residuals (signed distance of each face centroid from the plane)
        residuals = np.abs(face_centroids @ normal + d)
        # Keep `centered` for downstream debug if needed
        centered = face_centroids - centroid

        prim = {
            "patch_id": ids[0],
            "normal": normal.tolist(),
            "d": d,
            "centroid": centroid.tolist(),
            "face_count": int(len(face_idx)),
        }
        extent = _compute_mesh_bbox_size(mesh)
        surf = _build_infinite_plane(prim, extent)
        # Tag the source patches so the UI can track which segments produced this plane
        surf["source_patch_ids"] = ids
        # Quality metric: residual std along normal
        residuals = np.abs(centered @ normal)
        surf["fit_residual_mean"] = float(residuals.mean())
        surf["fit_residual_max"] = float(residuals.max())

        # Persist server-side so the intersection endpoint can use it
        plane_id = len(SESSION["user_planes"])
        SESSION["user_planes"].append({
            "id": plane_id,
            "normal": normal.tolist(),
            "d": d,
            "centroid": centroid.tolist(),
            "source_patch_ids": ids,
        })
        surf["plane_id"] = plane_id
        return surf
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/clear_user_planes")
def api_clear_user_planes():
    SESSION["user_planes"] = []
    return {"ok": True}


@app.post("/api/set_user_planes")
def api_set_user_planes(params: dict = {}):
    """Replace the entire user_planes list (used by undo/redo to resync state)."""
    planes = params.get("planes") or []
    SESSION["user_planes"] = [
        {
            "id": int(p.get("id", i)),
            "normal": list(p["normal"]),
            "d": float(p["d"]),
            "centroid": list(p.get("centroid", [0, 0, 0])),
            "source_patch_ids": list(p.get("source_patch_ids", [])),
        }
        for i, p in enumerate(planes)
    ]
    return {"ok": True, "n_planes": len(SESSION["user_planes"])}


@app.post("/api/intersect_user_planes")
def api_intersect_user_planes(params: dict = {}):
    """Pairwise plane-plane intersections of all user-created planes,
    each line clipped to the mesh AABB. Returns EdgeCurve[]."""
    try:
        import numpy as np
        planes = SESSION.get("user_planes") or []
        # Optional filter: only intersect among the given plane ids
        filter_ids = params.get("plane_ids")
        if filter_ids is not None:
            wanted = set(int(i) for i in filter_ids)
            planes = [p for p in planes if int(p["id"]) in wanted]
        if len(planes) < 2:
            return {"edges": [], "n_edges": 0}

        mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if mesh is None:
            raise HTTPException(400, "No mesh loaded")
        verts = np.asarray(mesh.vertices, dtype=float)
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        pad = 0.02 * float(np.linalg.norm(bbox_max - bbox_min))
        bbox_min = bbox_min - pad
        bbox_max = bbox_max + pad

        def clip(p0, dr):
            t_min, t_max = -np.inf, np.inf
            for ax in range(3):
                if abs(dr[ax]) < 1e-9:
                    if p0[ax] < bbox_min[ax] or p0[ax] > bbox_max[ax]:
                        return None
                    continue
                t1 = (bbox_min[ax] - p0[ax]) / dr[ax]
                t2 = (bbox_max[ax] - p0[ax]) / dr[ax]
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_min:
                    t_min = t1
                if t2 < t_max:
                    t_max = t2
                if t_min > t_max:
                    return None
            return p0 + t_min * dr, p0 + t_max * dr

        edges = []
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                pi, pj = planes[i], planes[j]
                ni = np.asarray(pi["normal"], dtype=float)
                nj = np.asarray(pj["normal"], dtype=float)
                di = float(pi["d"])
                dj = float(pj["d"])
                line_dir = np.cross(ni, nj)
                if np.linalg.norm(line_dir) < 1e-6:
                    continue  # parallel
                line_dir /= np.linalg.norm(line_dir)
                A = np.stack([ni, nj, line_dir])
                b = np.array([-di, -dj, 0.0])
                try:
                    p0 = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                clipped = clip(p0, line_dir)
                if clipped is None:
                    continue
                a, c = clipped
                edges.append({
                    "patch_a": int(pi["id"]),
                    "patch_b": int(pj["id"]),
                    "type_a": "user_plane",
                    "type_b": "user_plane",
                    "points": [a.tolist(), c.tolist()],
                    "n_points": 2,
                })
        return {"edges": edges, "n_edges": len(edges)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


def _point_in_poly_2d(pt, poly):
    x, y = float(pt[0]), float(pt[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)):
            xint = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def _stitch_boundary_loops(boundary_edges):
    """Stitch undirected edges into closed vertex loops."""
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    loops = []
    for start in list(adj.keys()):
        while adj[start]:
            loop = [start]
            cur = start
            prev = -1
            steps = 0
            max_steps = sum(len(v) for v in adj.values()) + 10
            while steps < max_steps:
                steps += 1
                candidates = [n for n in adj[cur] if n != prev]
                if not candidates:
                    break
                nxt = candidates[0]
                adj[cur].remove(nxt)
                adj[nxt].remove(cur)
                if nxt == start:
                    break
                loop.append(nxt)
                prev = cur
                cur = nxt
            if len(loop) >= 3:
                loops.append(loop)
    return loops


@app.post("/api/face_from_points")
def api_face_from_points(params: dict = {}):
    """Build a polygonal face whose corners are user-picked 3D points,
    projected onto a plane fitted from `patch_ids` (or fitted from the points
    themselves if no patch is given).

    Body: { "points": [[x,y,z], ...], "patch_ids": [int, ...] (optional) }
    """
    try:
        import numpy as np

        pts = np.asarray(params.get("points") or [], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
            raise HTTPException(400, "Need at least 3 points (x,y,z)")

        ids = [int(i) for i in (params.get("patch_ids") or [])]

        mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if mesh is None:
            raise HTTPException(400, "No mesh loaded")

        # Determine plane: prefer the patch fit so the face lies on the segment
        if ids and SESSION.get("labels") is not None:
            labels = SESSION["labels"]
            faces = np.asarray(mesh.faces)
            mask = np.isin(labels, np.array(ids, dtype=labels.dtype))
            face_idx = np.where(mask)[0]
            vert_idx = np.unique(faces[face_idx].flatten())
            patch_pts = np.asarray(mesh.vertices[vert_idx], dtype=float)
            centroid = patch_pts.mean(axis=0)
            centered = patch_pts - centroid
        else:
            centroid = pts.mean(axis=0)
            centered = pts - centroid

        cov = (centered.T @ centered) / max(len(centered) - 1, 1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]
        normal /= np.linalg.norm(normal)

        # Orient normal to roughly agree with the patch face normals
        if ids:
            try:
                avg_n = np.asarray(mesh.face_normals[face_idx]).mean(axis=0)
                if np.dot(avg_n, normal) < 0:
                    normal = -normal
            except Exception:
                pass

        # Plane basis
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1.0, 0.0, 0.0])
        else:
            u = np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)

        # Snap picked points onto plane
        rel = pts - centroid
        snapped = pts - np.outer(rel @ normal, normal)
        uv = np.column_stack([(snapped - centroid) @ u, (snapped - centroid) @ v])

        # Order points around the centroid by polar angle (creates simple polygon)
        # Works well for convex-ish quads/polygons; user clicks define the corners
        c2d = uv.mean(axis=0)
        angles = np.arctan2(uv[:, 1] - c2d[1], uv[:, 0] - c2d[0])
        order = np.argsort(angles)
        ordered_uv = uv[order]
        ordered_3d = snapped[order]

        # Fan triangulation from the first vertex (works for convex polygons,
        # which is what corner picks normally produce)
        triangles = []
        for i in range(1, len(ordered_uv) - 1):
            triangles.append([0, i, i + 1])

        return {
            "type": "trimmed_plane",
            "patch_ids": ids,
            "vertices": ordered_3d.astype(float).tolist(),
            "triangles": triangles,
            "boundary_indices": list(range(len(ordered_3d))),
            "normal": normal.tolist(),
            "centroid": centroid.tolist(),
            "u_axis": u.tolist(),
            "v_axis": v.tolist(),
            "n_boundary": int(len(ordered_3d)),
            "n_triangles": int(len(triangles)),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/face_from_patches")
def api_face_from_patches(params: dict = {}):
    """Build a polygonal face that follows the actual boundary of the selected
    patch(es), projected onto a fitted plane. Returns vertices + triangles.
    """
    try:
        import numpy as np
        from scipy.spatial import Delaunay

        ids = [int(i) for i in (params.get("patch_ids") or [])]
        if not ids:
            raise HTTPException(400, "patch_ids required")

        mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        labels = SESSION.get("labels")
        if mesh is None or labels is None:
            raise HTTPException(400, "Must segment mesh first")

        faces = np.asarray(mesh.faces)
        verts = np.asarray(mesh.vertices, dtype=float)
        mask = np.isin(labels, np.array(ids, dtype=labels.dtype))
        face_idx = np.where(mask)[0]
        if len(face_idx) < 3:
            raise HTTPException(400, "Not enough faces in selected patches")

        patch_faces = faces[face_idx]  # (M, 3)

        # Boundary edges: appear in exactly one patch face when sorted
        e = np.concatenate(
            [patch_faces[:, [0, 1]], patch_faces[:, [1, 2]], patch_faces[:, [2, 0]]],
            axis=0,
        )
        e_sorted = np.sort(e, axis=1)
        uniq, counts = np.unique(e_sorted, axis=0, return_counts=True)
        boundary_edges = uniq[counts == 1]
        if len(boundary_edges) < 3:
            raise HTTPException(400, "Patch has no usable boundary")

        loops = _stitch_boundary_loops(boundary_edges.tolist())
        if not loops:
            raise HTTPException(400, "Could not stitch boundary loops")

        # Fit plane via PCA on patch vertices
        vert_idx = np.unique(patch_faces.flatten())
        pts = verts[vert_idx]
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = (centered.T @ centered) / max(len(pts) - 1, 1)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0]
        normal /= np.linalg.norm(normal)
        try:
            avg_n = np.asarray(mesh.face_normals[face_idx]).mean(axis=0)
            if np.dot(avg_n, normal) < 0:
                normal = -normal
        except Exception:
            pass

        # Plane basis
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1.0, 0.0, 0.0])
        else:
            u = np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)

        def project_to_plane(p3d):
            rel = p3d - centroid
            snapped = p3d - np.outer(rel @ normal, normal)
            uv = np.column_stack([(snapped - centroid) @ u, (snapped - centroid) @ v])
            return snapped, uv

        # Largest loop = outer boundary; rest treated as holes (simple approach: ignored
        # for triangle filtering since Delaunay-on-outer + centroid test only uses outer)
        loops.sort(key=len, reverse=True)
        outer = loops[0]
        outer_pts3d = verts[outer]
        outer_snapped, outer_uv = project_to_plane(outer_pts3d)

        if len(outer_uv) < 3:
            raise HTTPException(400, "Outer loop too small")

        # Constrained-ish triangulation: Delaunay on outer points, keep only
        # triangles whose centroid lies inside the polygon.
        try:
            tri = Delaunay(outer_uv)
        except Exception as exc:
            raise HTTPException(400, f"Delaunay failed: {exc}")

        kept = []
        for simplex in tri.simplices:
            c2d = outer_uv[simplex].mean(axis=0)
            if _point_in_poly_2d(c2d, outer_uv):
                kept.append(simplex.tolist())

        if not kept:
            raise HTTPException(400, "Triangulation produced no interior faces")

        # Compute residuals for fit quality
        residuals = np.abs(centered @ normal)

        return {
            "type": "trimmed_plane",
            "patch_ids": ids,
            "vertices": outer_snapped.astype(float).tolist(),
            "triangles": kept,
            "boundary_indices": list(range(len(outer))),
            "normal": normal.tolist(),
            "centroid": centroid.tolist(),
            "u_axis": u.tolist(),
            "v_axis": v.tolist(),
            "n_boundary": len(outer),
            "n_triangles": len(kept),
            "fit_residual_mean": float(residuals.mean()),
            "fit_residual_max": float(residuals.max()),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


POINT2CYL_PY = r"C:\Users\Dali Design\miniconda3\envs\point2cyl\python.exe"
POINT2CYL_RUNNER = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "external", "point2cyl", "run_inference.py")
)


@app.post("/api/run_point2cyl")
def api_run_point2cyl(params: dict = {}):
    """Run Point2Cyl inference on the loaded mesh.

    Spawns a subprocess in the dedicated `point2cyl` conda env so the model's
    PyTorch + torch-scatter dependencies are isolated from this server.
    Returns a list of detected extrusion segments (axis, center, radius, extent).
    """
    try:
        import subprocess
        import json as _json

        mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if mesh is None:
            raise HTTPException(400, "Load a mesh first")

        # Write the current mesh to a temp STL the runner can read
        tmp_stl = os.path.join(SESSION["temp_dir"], "point2cyl_input.stl")
        mesh.export(tmp_stl)

        out_json = os.path.join(SESSION["temp_dir"], "point2cyl_output.json")
        if os.path.exists(out_json):
            try:
                os.remove(out_json)
            except Exception:
                pass

        if not os.path.exists(POINT2CYL_PY):
            raise HTTPException(500, f"Point2Cyl python not found at {POINT2CYL_PY}")
        if not os.path.exists(POINT2CYL_RUNNER):
            raise HTTPException(500, f"Point2Cyl runner not found at {POINT2CYL_RUNNER}")

        proc = subprocess.run(
            [POINT2CYL_PY, POINT2CYL_RUNNER, tmp_stl, out_json],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise HTTPException(500, f"Point2Cyl failed: {proc.stderr or proc.stdout}")

        if not os.path.exists(out_json):
            raise HTTPException(500, f"Point2Cyl produced no output: {proc.stdout}")

        with open(out_json, "r") as f:
            result = _json.load(f)

        SESSION["point2cyl_result"] = result
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/segment")
def api_segment(params: dict = {}):
    try:
        from pipeline.segmentation import segment_mesh
        # Reset merge groups when re-segmenting (patch ids change)
        SESSION["merge_parent"] = {}
        result = segment_mesh(params, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/fit_primitives")
def api_fit_primitives():
    try:
        from pipeline.primitive_fitting import fit_primitives
        result = fit_primitives({}, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/detect_features")
def api_detect_features():
    try:
        from pipeline.feature_detection import detect_features
        result = detect_features({}, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/reconstruct_brep")
def api_reconstruct_brep():
    try:
        from pipeline.brep_reconstruction import reconstruct_brep
        result = reconstruct_brep({}, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/build_polyhedral_cad")
def api_build_polyhedral_cad(params: dict = {}):
    """Return the low-poly polyhedral CAD as JSON for live rendering.
    Uses snap-to-plane + quadric decimation for a watertight low-poly mesh."""
    try:
        from cad.mesh_export import _build_watertight_snapped_mesh
        import numpy as np
        verts, faces = _build_watertight_snapped_mesh(SESSION, defeature_fillets=True)
        target_faces = int(params.get("target_faces", 2000))
        try:
            import open3d as o3d
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.remove_duplicated_vertices()
            o3d_mesh.remove_duplicated_triangles()
            o3d_mesh.remove_degenerate_triangles()
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            verts = np.asarray(o3d_mesh.vertices)
            faces = np.asarray(o3d_mesh.triangles)
        except Exception:
            pass
        return {
            "vertices": verts.astype(float).tolist(),
            "faces": faces.astype(int).tolist(),
            "n_vertices": int(len(verts)),
            "n_faces": int(len(faces)),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/build_cad_preview")
def api_build_cad_preview(params: dict = {}):
    try:
        from pipeline.cad_preview import build_cad_preview
        result = build_cad_preview(params, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/intersect_surfaces")
def api_intersect_surfaces(params: dict = {}):
    try:
        from pipeline.intersection import intersect_surfaces
        result = intersect_surfaces(params, progress_callback=progress_noop, session=SESSION)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/export_step")
def api_export_step(params: dict = {}):
    try:
        from cad.export import export_step
        output_path = params.get("output_path",
                                  os.path.join(SESSION["temp_dir"], "output.step"))
        result = export_step(
            {"output_path": output_path},
            progress_callback=progress_noop,
            session=SESSION,
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/download_step")
def download_step():
    path = os.path.join(SESSION["temp_dir"], "output.step")
    if not os.path.exists(path):
        raise HTTPException(404, "No STEP file exported yet")
    return FileResponse(path, filename="output.step", media_type="application/octet-stream")


@app.post("/api/export_cad")
def api_export_cad(params: dict = {}):
    """Tessellated export of CAD preview (no pythonocc needed). Format: stl|obj|ply."""
    try:
        from cad.mesh_export import export_cad_preview
        fmt = (params.get("format") or "stl").lower()
        output_path = params.get("output_path",
                                  os.path.join(SESSION["temp_dir"], f"output.{fmt}"))
        result = export_cad_preview(
            {"format": fmt, "output_path": output_path},
            progress_callback=progress_noop,
            session=SESSION,
        )
        # Make filename available for download
        result["filename"] = os.path.basename(result["output_path"])
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/download_cad/{filename}")
def download_cad(filename: str):
    path = os.path.join(SESSION["temp_dir"], filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


@app.get("/api/transfer/{filename}")
def get_transfer_file(filename: str):
    """Serve binary transfer files (mesh data, labels) to the browser."""
    path = os.path.join(SESSION["temp_dir"], filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {filename}")
    return FileResponse(path, media_type="application/octet-stream")


if __name__ == "__main__":
    print("Starting Geomagic Claude backend on http://localhost:8321")
    uvicorn.run(app, host="0.0.0.0", port=8321)

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
from mesh_io_pkg.serialization import mesh_to_transfer_file

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
    "temp_dir": tempfile.mkdtemp(prefix="geomagic_"),
    # Phase E0 — mechanical intent reconstruction state. Single object,
    # NOT a bag of session keys. See pipeline.reconstruction.state.
    "recon_state": None,
}


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

        progress_noop("load", 10, "Loading mesh file...")
        mesh = load_mesh_file(load_path)
        mesh_id = str(uuid.uuid4())[:8]

        progress_noop("load", 50, "Preparing transfer file...")
        transfer_path = mesh_to_transfer_file(mesh, SESSION["temp_dir"], mesh_id)

        SESSION["mesh"] = mesh
        SESSION["mesh_id"] = mesh_id
        SESSION["preprocessed"] = None
        SESSION["recon_state"] = None

        progress_noop("load", 100, "Mesh loaded")

        return {
            "mesh_id": mesh_id,
            "transfer_path": transfer_path,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds_min": mesh.bounds[0].tolist(),
            "bounds_max": mesh.bounds[1].tolist(),
            "is_watertight": bool(mesh.is_watertight),
        }
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


@app.post("/api/intent/export_cad_geometry")
def api_export_cad_geometry(params: dict = {}):
    """Build clean CAD geometry from detected surfaces and export.

    Each plane region → flat polygon, each cylinder → smooth patch.
    Returns transfer file URL for viewport + exports to disk.
    """
    try:
        from pipeline.reconstruction.cad_export import build_cad_geometry
        state = SESSION.get("recon_state")
        if state is None:
            raise HTTPException(400, "No reconstruction state — run /api/intent/run first")
        mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if mesh is None:
            raise HTTPException(400, "No mesh loaded")

        progress_noop("cad", 30, "Building CAD geometry from surfaces...")
        cad_mesh, stats = build_cad_geometry(state, mesh)

        # Export as STL and OBJ to temp dir
        fmt = params.get("format", "stl").lower()
        output_path = os.path.join(SESSION["temp_dir"], f"cad_output.{fmt}")
        cad_mesh.export(output_path)

        # Also save transfer file for viewport
        cad_id = f"{state.mesh_id}_cad"
        transfer_path = mesh_to_transfer_file(cad_mesh, SESSION["temp_dir"], cad_id)

        progress_noop("cad", 100, "CAD geometry exported")
        return {
            **stats,
            "transfer_path": transfer_path,
            "output_path": output_path,
            "filename": os.path.basename(output_path),
            "format": fmt,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/intent/export_step")
def api_intent_export_step(params: dict = {}):
    """Build OCC B-Rep and export as STEP file.

    Builds analytic plane faces + cylinder/cone primitives from E2 trimmed
    face data. Produces true CAD geometry, not a tessellated mesh.
    """
    try:
        from pipeline.reconstruction.brep import build_step_from_trimmed_faces
        state = SESSION.get("recon_state")
        if state is None:
            raise HTTPException(400, "No reconstruction state — run /api/intent/run first")
        if not state.trimmed_faces:
            raise HTTPException(400, "No trimmed faces — run /api/intent/trim first")

        progress_noop("step", 10, "Building analytic B-Rep from trimmed faces...")
        result = build_step_from_trimmed_faces(state, SESSION["temp_dir"])
        progress_noop("step", 100, "STEP export complete")

        # Store step path for download
        SESSION["step_path"] = result.get("step_path")
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/intent/download_step")
def api_download_step():
    """Download the exported STEP file."""
    path = SESSION.get("step_path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "No STEP file — run /api/intent/export_step first")
    return FileResponse(path, filename="scan_to_cad.step", media_type="application/octet-stream")


@app.get("/api/intent/reconstruct_live")
async def api_reconstruct_live(api_key: str = None, use_ai: bool = True):
    """Stream live reconstruction: one SSE event per region.

    Each event contains the AI classification, new geometry, and deviation.
    The frontend renders surfaces as they arrive in real-time.
    """
    from fastapi.responses import StreamingResponse
    from pipeline.reconstruction.reconstruct_live import reconstruct_live

    state = SESSION.get("recon_state")
    if state is None:
        raise HTTPException(400, "No reconstruction state")
    mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
    if mesh is None:
        raise HTTPException(400, "No mesh")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def event_stream():
        import json
        for event in reconstruct_live(state, mesh, api_key=key, use_ai=use_ai):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/intent/classify")
def api_intent_classify(params: dict = {}):
    """AI-powered surface classification using Claude API.

    Analyzes each region's geometry and classifies it as PLANE, CYLINDER,
    CONE, SPHERE, FILLET, CHAMFER, FREEFORM, or UNKNOWN with reasoning.
    """
    try:
        from pipeline.reconstruction.ai_classify import classify_regions
        state = SESSION.get("recon_state")
        if state is None:
            raise HTTPException(400, "No reconstruction state — run /api/intent/run first")
        mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if mesh is None:
            raise HTTPException(400, "No mesh loaded")

        progress_noop("classify", 30, "Classifying regions with AI...")
        result = classify_regions(
            state, mesh,
            api_key=params.get("api_key"),
            part_description=params.get("part_description", ""),
        )
        progress_noop("classify", 100, f"{result['n_classified']} regions classified")
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/intent/deviation")
def api_intent_deviation(params: dict = {}):
    """Compute deviation analysis between original mesh and fitted surfaces."""
    try:
        from pipeline.reconstruction.deviation import compute_deviation_analysis
        state = SESSION.get("recon_state")
        if state is None:
            raise HTTPException(400, "No reconstruction state — run /api/intent/run first")
        if state.snap_result is None:
            raise HTTPException(400, "No snap result — run /api/intent/snap first")
        original_mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if original_mesh is None:
            raise HTTPException(400, "No mesh loaded")

        progress_noop("deviation", 50, "Computing deviation analysis...")
        result = compute_deviation_analysis(state, original_mesh)
        progress_noop("deviation", 100, "Deviation analysis complete")
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/intent/trim")
def api_intent_trim(params: dict = {}):
    """Construct trimmed faces for all HIGH-fit regions.

    Extracts boundary loops, computes UV parameterization, and stores
    TrimmedFace objects on the reconstruction state.
    """
    try:
        from pipeline.reconstruction.trim import construct_trimmed_faces, get_trim_summary
        state = SESSION.get("recon_state")
        if state is None:
            raise HTTPException(400, "No reconstruction state — run /api/intent/run first")
        full_mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if full_mesh is None:
            raise HTTPException(400, "No mesh loaded")

        progress_noop("trim", 10, "Constructing trimmed faces...")
        trimmed = construct_trimmed_faces(state, full_mesh)
        progress_noop("trim", 100, f"{len(trimmed)} trimmed faces constructed")

        summary = get_trim_summary(state)
        # Include boundary loop data for frontend visualization
        boundary_data = []
        for tf in trimmed.values():
            loops = [tf.outer_loop.to_dict()]
            for il in tf.inner_loops:
                loops.append(il.to_dict())
            boundary_data.append({
                "region_id": tf.region_id,
                "surface_type": tf.surface_type,
                "loops": loops,
            })

        return {
            **summary,
            "faces": boundary_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/intent/snap")
def api_intent_snap(params: dict = {}):
    """Snap all HIGH-region vertices to their analytic surfaces.

    Returns a transfer file URL for the snapped mesh and snap statistics.
    """
    try:
        import trimesh
        from pipeline.reconstruction.snap import snap_mesh_to_analytics
        state = SESSION.get("recon_state")
        if state is None:
            raise HTTPException(400, "No reconstruction state — run /api/intent/run first")
        full_mesh = SESSION.get("preprocessed") or SESSION.get("mesh")
        if full_mesh is None:
            raise HTTPException(400, "No mesh loaded")

        progress_noop("snap", 10, "Snapping vertices to analytic surfaces...")
        snap_result = snap_mesh_to_analytics(state, full_mesh)
        state.snap_result = snap_result

        progress_noop("snap", 80, "Serializing snapped mesh...")
        snapped_mesh = trimesh.Trimesh(
            vertices=snap_result.snapped_vertices,
            faces=snap_result.faces,
            process=False,
        )
        snap_id = f"{state.mesh_id}_snapped"
        transfer_path = mesh_to_transfer_file(snapped_mesh, SESSION["temp_dir"], snap_id)

        progress_noop("snap", 100, "Snap complete")
        return {
            "transfer_path": transfer_path,
            "snap_id": snap_id,
            **snap_result.stats,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


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

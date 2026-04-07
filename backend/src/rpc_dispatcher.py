"""Routes JSON-RPC method calls to pipeline handlers."""
import os
import uuid
import tempfile

# Session state — persists across calls within one server lifetime
SESSION = {
    "mesh": None,           # trimesh.Trimesh
    "mesh_id": None,
    "preprocessed": None,
    "labels": None,         # numpy array of per-face labels
    "patches": None,        # list of patch dicts
    "primitives": None,     # list of fitted primitive dicts
    "features": None,       # list of detected features
    "temp_dir": tempfile.mkdtemp(prefix="geomagic_"),
}


class RPCDispatcher:
    def __init__(self):
        self._methods = {}
        self._register_methods()

    def _register_methods(self):
        from pipeline.preprocessing import load_mesh, preprocess_mesh
        from pipeline.segmentation import segment_mesh
        from pipeline.primitive_fitting import fit_primitives
        from pipeline.feature_detection import detect_features
        from pipeline.intersection import intersect_surfaces
        from pipeline.brep_reconstruction import reconstruct_brep
        from cad.export import export_step
        from mesh_io_pkg.mesh_io import get_mesh_info

        self._methods = {
            "ping": self._ping,
            "load_mesh": load_mesh,
            "get_mesh_info": get_mesh_info,
            "preprocess": preprocess_mesh,
            "segment": segment_mesh,
            "fit_primitives": fit_primitives,
            "detect_features": detect_features,
            "intersect_surfaces": intersect_surfaces,
            "reconstruct_brep": reconstruct_brep,
            "export_step": export_step,
        }

    def _ping(self, params, progress_callback=None, session=None):
        return {"status": "ok", "version": "0.1.0"}

    def dispatch(self, method: str, params: dict, progress_callback=None):
        if method not in self._methods:
            raise ValueError(f"Unknown method: {method}")
        handler = self._methods[method]
        return handler(params, progress_callback=progress_callback, session=SESSION)

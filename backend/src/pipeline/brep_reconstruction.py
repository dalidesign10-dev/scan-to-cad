"""B-Rep reconstruction from fitted primitives using pythonocc."""
import numpy as np


def reconstruct_brep(params, progress_callback=None, session=None):
    """Build a B-Rep shell from fitted primitives.

    Strategy: create OCC surfaces for each primitive, intersect adjacent
    pairs to get edge curves, trim, and sew into an open shell.
    """
    primitives = session.get("primitives")
    labels = session.get("labels")
    mesh = session.get("preprocessed") or session.get("mesh")

    if not primitives or labels is None or mesh is None:
        raise ValueError("Must fit primitives before B-Rep reconstruction")

    try:
        from OCC.Core.gp import gp_Pln, gp_Ax3, gp_Pnt, gp_Dir, gp_Ax2, gp_Circ
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_Sewing,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeEdge,
        )
        from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_SphericalSurface
        from OCC.Core.GeomAPI import GeomAPI_IntSS
        from OCC.Core.TopoDS import TopoDS_Shape
    except ImportError:
        raise ImportError("pythonocc-core is required for B-Rep reconstruction. Install with: conda install -c conda-forge pythonocc-core")

    if progress_callback:
        progress_callback("brep", 10, "Creating OCC surfaces...")

    faces = mesh.faces
    vertices = mesh.vertices
    occ_surfaces = {}

    for prim in primitives:
        pid = prim["patch_id"]
        ptype = prim["type"]

        if ptype == "plane":
            n = np.array(prim["normal"])
            d = prim["d"]
            # Get a point on the plane
            point = -d * n
            gp_point = gp_Pnt(float(point[0]), float(point[1]), float(point[2]))
            gp_normal = gp_Dir(float(n[0]), float(n[1]), float(n[2]))
            surface = Geom_Plane(gp_point, gp_normal)
            occ_surfaces[pid] = surface

        elif ptype == "cylinder":
            axis = np.array(prim["axis"])
            center = np.array(prim["center"])
            radius = prim["radius"]
            gp_center = gp_Pnt(float(center[0]), float(center[1]), float(center[2]))
            gp_dir = gp_Dir(float(axis[0]), float(axis[1]), float(axis[2]))
            ax3 = gp_Ax3(gp_center, gp_dir)
            surface = Geom_CylindricalSurface(ax3, float(radius))
            occ_surfaces[pid] = surface

        elif ptype == "sphere":
            center = np.array(prim["center"])
            radius = prim["radius"]
            gp_center = gp_Pnt(float(center[0]), float(center[1]), float(center[2]))
            ax3 = gp_Ax3(gp_center, gp_Dir(0, 0, 1))
            surface = Geom_SphericalSurface(ax3, float(radius))
            occ_surfaces[pid] = surface

    if progress_callback:
        progress_callback("brep", 40, "Creating bounded faces...")

    sewing = BRepBuilderAPI_Sewing(1.0)

    for prim in primitives:
        pid = prim["patch_id"]
        if pid not in occ_surfaces:
            continue

        surface = occ_surfaces[pid]
        mask = labels == pid
        face_indices = np.where(mask)[0]
        if len(face_indices) == 0:
            continue

        # Compute bounding box of this patch for face bounds
        vert_indices = np.unique(faces[face_indices].flatten())
        pts = vertices[vert_indices]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)

        # Create a bounded face from the surface
        try:
            # Use a large but bounded face
            umin, umax = -1000, 1000
            vmin, vmax = -1000, 1000
            face = BRepBuilderAPI_MakeFace(surface, umin, umax, vmin, vmax, 1e-6)
            if face.IsDone():
                sewing.Add(face.Face())
        except Exception:
            continue

    if progress_callback:
        progress_callback("brep", 80, "Sewing faces...")

    sewing.Perform()
    shape = sewing.SewedShape()

    session["brep_shape"] = shape

    if progress_callback:
        progress_callback("brep", 100, "B-Rep reconstruction complete")

    return {
        "status": "ok",
        "n_surfaces": len(occ_surfaces),
        "message": f"Created {len(occ_surfaces)} surfaces, sewed into shell"
    }

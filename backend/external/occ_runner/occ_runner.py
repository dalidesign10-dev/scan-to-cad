"""Standalone OCC operations runner.

Designed to be invoked by the main backend via subprocess. Reads a JSON
request from stdin and writes a JSON response to stdout.

Request shapes:
    {"op": "build", "mesh_path": "...", "target_faces": 1500}
    {"op": "fillet", "shape_path": "...", "radius": 0.5, "min_dihedral_deg": 30}
    {"op": "chamfer", "shape_path": "...", "distance": 0.5, "min_dihedral_deg": 30}

The build op also persists the OCC shape to a .brep file (via BRepTools)
so subsequent fillet/chamfer ops can reload it without re-decimating.
"""
import sys
import os
import json
import traceback
import tempfile
import numpy as np
import trimesh

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Pln
from OCC.Core.Geom import Geom_CylindricalSurface, Geom_ConicalSurface
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
)
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SHELL, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps


# ----------------------------------------------------------------------------
# Mesh -> polyhedral solid
# ----------------------------------------------------------------------------
def build_polyhedral_solid_from_mesh(mesh: trimesh.Trimesh):
    sewing = BRepBuilderAPI_Sewing(1.0e-3)

    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.faces, dtype=int)

    n_added = 0
    for tri in tris:
        a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        edge_ab = b - a
        edge_ac = c - a
        cross = np.cross(edge_ab, edge_ac)
        if np.linalg.norm(cross) < 1e-10:
            continue
        try:
            poly = BRepBuilderAPI_MakePolygon()
            poly.Add(gp_Pnt(float(a[0]), float(a[1]), float(a[2])))
            poly.Add(gp_Pnt(float(b[0]), float(b[1]), float(b[2])))
            poly.Add(gp_Pnt(float(c[0]), float(c[1]), float(c[2])))
            poly.Close()
            wire = poly.Wire()
            face_builder = BRepBuilderAPI_MakeFace(wire, True)
            if face_builder.IsDone():
                sewing.Add(face_builder.Face())
                n_added += 1
        except Exception:
            continue

    if n_added == 0:
        raise RuntimeError("No valid triangles for polyhedral B-Rep")

    sewing.Perform()
    sewed = sewing.SewedShape()

    explorer = TopExp_Explorer(sewed, TopAbs_SHELL)
    if not explorer.More():
        # Return the sewed compound — fillet may still work
        return sewed
    shell = topods.Shell(explorer.Current())

    try:
        sb = BRepBuilderAPI_MakeSolid(shell)
        if sb.IsDone():
            return sb.Solid()
    except Exception:
        pass
    return shell


# ----------------------------------------------------------------------------
# Tessellation
# ----------------------------------------------------------------------------
def tessellate_shape(shape, deflection: float = 0.5):
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)

    all_verts = []
    all_faces = []
    vert_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is not None:
            trsf = location.Transformation()
            n_nodes = triangulation.NbNodes()
            for i in range(1, n_nodes + 1):
                p = triangulation.Node(i)
                p.Transform(trsf)
                all_verts.append([p.X(), p.Y(), p.Z()])
            n_tri = triangulation.NbTriangles()
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for i in range(1, n_tri + 1):
                t = triangulation.Triangle(i)
                n1, n2, n3 = t.Get()
                if reversed_face:
                    all_faces.append([
                        vert_offset + n1 - 1,
                        vert_offset + n3 - 1,
                        vert_offset + n2 - 1,
                    ])
                else:
                    all_faces.append([
                        vert_offset + n1 - 1,
                        vert_offset + n2 - 1,
                        vert_offset + n3 - 1,
                    ])
            vert_offset += n_nodes
        explorer.Next()

    if not all_verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.asarray(all_verts, dtype=float), np.asarray(all_faces, dtype=int)


# ----------------------------------------------------------------------------
# Sharp edge collection
# ----------------------------------------------------------------------------
def collect_sharp_edges(shape, min_dihedral_deg: float):
    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    sharp_edges = []
    n = edge_to_faces.Size()
    for i in range(1, n + 1):
        edge = topods.Edge(edge_to_faces.FindKey(i))
        face_list = edge_to_faces.FindFromIndex(i)
        if face_list.Size() != 2:
            continue
        face_a = topods.Face(face_list.First())
        face_b = topods.Face(face_list.Last())
        normals = []
        for face in (face_a, face_b):
            try:
                surf = BRep_Tool.Surface(face)
                ad = BRepAdaptor_Surface(face)
                u = 0.5 * (ad.FirstUParameter() + ad.LastUParameter())
                v = 0.5 * (ad.FirstVParameter() + ad.LastVParameter())
                props = GeomLProp_SLProps(surf, u, v, 1, 1e-6)
                if not props.IsNormalDefined():
                    normals.append(None)
                    continue
                gp_n = props.Normal()
                nx, ny, nz = gp_n.X(), gp_n.Y(), gp_n.Z()
                if face.Orientation() == TopAbs_REVERSED:
                    nx, ny, nz = -nx, -ny, -nz
                normals.append(np.array([nx, ny, nz]))
            except Exception:
                normals.append(None)
        if any(x is None for x in normals):
            continue
        n1, n2 = normals
        cos_a = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_a)))
        if angle_deg > min_dihedral_deg:
            sharp_edges.append(edge)
    return sharp_edges


# ----------------------------------------------------------------------------
# Persist / load shapes
# ----------------------------------------------------------------------------
def write_brep(shape, path):
    breptools.Write(shape, path)


def read_brep(path):
    builder = BRep_Builder()
    shape = TopoDS_Shape()
    breptools.Read(shape, path, builder)
    return shape


# ----------------------------------------------------------------------------
# Op dispatchers
# ----------------------------------------------------------------------------
def op_build(req):
    mesh_path = req["mesh_path"]
    out_brep = req["out_brep"]
    debug_log = out_brep + ".log"

    def trace(msg):
        with open(debug_log, "a") as f:
            f.write(msg + "\n")

    trace(f"START load mesh_path={mesh_path}")
    # NOTE: trimesh 4.11.5 in this env crashes natively when force='mesh' is
    # passed. Load without it and merge any sub-geometries manually.
    loaded = trimesh.load(mesh_path)
    if isinstance(loaded, trimesh.Scene):
        merged = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
        mesh = merged
    else:
        mesh = loaded
    trace(f"loaded mesh: verts={len(mesh.vertices)} faces={len(mesh.faces)}")

    trace("calling build_polyhedral_solid_from_mesh")
    shape = build_polyhedral_solid_from_mesh(mesh)
    trace("build_polyhedral_solid_from_mesh OK")

    trace("write_brep")
    write_brep(shape, out_brep)
    trace("write_brep OK")

    trace("tessellate_shape")
    verts, faces = tessellate_shape(shape, deflection=0.5)
    trace(f"tessellate OK verts={len(verts)} faces={len(faces)}")
    return {
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "vertices": verts.astype(float).tolist(),
        "faces": faces.astype(int).tolist(),
        "decimated_faces": int(len(mesh.faces)),
        "brep_path": out_brep,
    }


def op_fillet_or_chamfer(req, op_kind):
    shape = read_brep(req["shape_path"])
    out_brep = req["out_brep"]
    radius_or_distance = float(req["radius"] if op_kind == "fillet" else req["distance"])
    min_dihedral = float(req.get("min_dihedral_deg", 20.0))

    sharp = collect_sharp_edges(shape, min_dihedral)
    if not sharp:
        raise RuntimeError(f"No edges found with dihedral > {min_dihedral}°")

    if op_kind == "fillet":
        builder = BRepFilletAPI_MakeFillet(shape)
        for e in sharp:
            try:
                builder.Add(radius_or_distance, e)
            except Exception:
                continue
    else:
        builder = BRepFilletAPI_MakeChamfer(shape)
        for e in sharp:
            try:
                builder.Add(radius_or_distance, e)
            except Exception:
                continue

    builder.Build()
    if not builder.IsDone():
        raise RuntimeError(f"OCC {op_kind} failed to build")
    new_shape = builder.Shape()
    write_brep(new_shape, out_brep)
    verts, faces = tessellate_shape(new_shape, deflection=0.5)
    return {
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "vertices": verts.astype(float).tolist(),
        "faces": faces.astype(int).tolist(),
        "n_edges_modified": int(len(sharp)),
        "brep_path": out_brep,
    }


def op_export(req):
    """Export the current B-Rep to STL / STEP / BREP based on `format`."""
    shape = read_brep(req["shape_path"])
    out_path = req["output_path"]
    fmt = req.get("format", "stl").lower()

    if fmt == "stl":
        from OCC.Extend.DataExchange import write_stl_file
        BRepMesh_IncrementalMesh(shape, 0.3, False, 0.5, True)
        # Suppress OCC's "file already exists" warnings on stdout
        import os as _os
        sys.stdout.flush()
        saved_fd = _os.dup(1)
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        try:
            _os.dup2(devnull_fd, 1)
            # Pre-delete to avoid the warning entirely
            if _os.path.exists(out_path):
                _os.remove(out_path)
            write_stl_file(shape, out_path, mode="binary", linear_deflection=0.3, angular_deflection=0.5)
        finally:
            _os.dup2(saved_fd, 1)
            _os.close(saved_fd)
            _os.close(devnull_fd)
    elif fmt == "step":
        # Redirect FD 1 (stdout) at the OS level so OCC's printf doesn't
        # corrupt our JSON channel. OCC is a C++ library so Python-level
        # sys.stdout reassignment isn't enough.
        import os as _os
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IFSelect import IFSelect_RetDone
        sys.stdout.flush()
        saved_fd = _os.dup(1)
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        try:
            _os.dup2(devnull_fd, 1)
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_AsIs)
            status = writer.Write(out_path)
        finally:
            _os.dup2(saved_fd, 1)
            _os.close(saved_fd)
            _os.close(devnull_fd)
        if status != IFSelect_RetDone:
            raise RuntimeError("STEP export failed")
    elif fmt == "brep":
        write_brep(shape, out_path)
    else:
        raise ValueError(f"Unknown export format: {fmt}")

    return {"output_path": out_path, "format": fmt, "size": os.path.getsize(out_path)}


def op_build_trimmed(req):
    """Build analytic B-Rep faces from trimmed face data and export as STEP.

    Input: {
        "op": "build_trimmed",
        "faces": [{
            "surface_type": "plane"|"cylinder"|"cone",
            "surface_params": {...},
            "outer_loop": [[x,y,z], ...],
            "inner_loops": [[[x,y,z], ...], ...]
        }, ...],
        "out_brep": "/path/to/output.brep",
        "out_step": "/path/to/output.step"
    }
    """
    import math as _math
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
    from OCC.Core.gp import gp_Ax2
    from OCC.Core.TopoDS import TopoDS_Compound

    faces_data = req["faces"]
    out_brep = req["out_brep"]
    out_step = req["out_step"]

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    n_built = 0
    n_failed = 0
    MIN_EDGE_LEN = 1e-4

    for i, fd in enumerate(faces_data):
        try:
            stype = fd["surface_type"]
            params = fd["surface_params"]
            outer_loop = fd["outer_loop"]

            if stype == "plane":
                # Build plane face from boundary wire
                normal = params["normal"]
                d_val = params.get("d", 0.0)
                if "centroid" in params:
                    pt = params["centroid"]
                else:
                    pt = [-d_val * normal[0], -d_val * normal[1], -d_val * normal[2]]
                surface = gp_Pln(
                    gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])),
                    gp_Dir(float(normal[0]), float(normal[1]), float(normal[2])),
                )
                wire_builder = BRepBuilderAPI_MakeWire()
                ne = 0
                for j in range(len(outer_loop)):
                    a = outer_loop[j]
                    b = outer_loop[(j + 1) % len(outer_loop)]
                    dx = float(b[0]-a[0]); dy = float(b[1]-a[1]); dz = float(b[2]-a[2])
                    if (dx*dx + dy*dy + dz*dz) < MIN_EDGE_LEN * MIN_EDGE_LEN:
                        continue
                    me = BRepBuilderAPI_MakeEdge(
                        gp_Pnt(float(a[0]), float(a[1]), float(a[2])),
                        gp_Pnt(float(b[0]), float(b[1]), float(b[2])),
                    )
                    if me.IsDone():
                        wire_builder.Add(me.Edge())
                        ne += 1
                if ne < 3:
                    raise ValueError(f"Too few edges ({ne})")
                face = BRepBuilderAPI_MakeFace(surface, wire_builder.Wire(), True)
                if not face.IsDone():
                    raise RuntimeError("MakeFace failed")
                builder.Add(compound, face.Face())
                n_built += 1

            elif stype == "cylinder":
                # Use BRepPrimAPI for a proper cylinder solid
                center = [float(x) for x in params["center"]]
                axis = [float(x) for x in params["axis"]]
                radius = float(params["radius"])
                # Compute height from boundary loop extent along axis
                import numpy as _np
                pts = _np.array(outer_loop, dtype=float)
                projections = (pts - center) @ _np.array(axis)
                h_min, h_max = float(projections.min()), float(projections.max())
                height = max(h_max - h_min, 0.1)
                # Shift center to bottom of cylinder
                base = [center[k] + h_min * axis[k] for k in range(3)]
                ax2 = gp_Ax2(
                    gp_Pnt(base[0], base[1], base[2]),
                    gp_Dir(axis[0], axis[1], axis[2]),
                )
                cyl = BRepPrimAPI_MakeCylinder(ax2, radius, height)
                builder.Add(compound, cyl.Shape())
                n_built += 1

            elif stype == "cone":
                # Use BRepPrimAPI for a proper cone solid
                apex = [float(x) for x in params["apex"]]
                axis = [float(x) for x in params["axis"]]
                if "half_angle_rad" in params:
                    half_angle = float(params["half_angle_rad"])
                elif "half_angle_deg" in params:
                    half_angle = _math.radians(float(params["half_angle_deg"]))
                else:
                    raise KeyError("No half_angle in cone params")
                # Compute extent from boundary loop
                import numpy as _np
                pts = _np.array(outer_loop, dtype=float)
                projections = (pts - apex) @ _np.array(axis)
                h_min, h_max = float(projections.min()), float(projections.max())
                if h_min < 0.01:
                    h_min = 0.01  # avoid apex singularity
                r1 = h_min * _math.tan(half_angle)
                r2 = h_max * _math.tan(half_angle)
                height = h_max - h_min
                if height < 0.01 or r1 < 0.001 or r2 < 0.001:
                    raise ValueError(f"Degenerate cone: h={height:.3f} r1={r1:.3f} r2={r2:.3f}")
                base = [apex[k] + h_min * axis[k] for k in range(3)]
                ax2 = gp_Ax2(
                    gp_Pnt(base[0], base[1], base[2]),
                    gp_Dir(axis[0], axis[1], axis[2]),
                )
                cone = BRepPrimAPI_MakeCone(ax2, r1, r2, height)
                builder.Add(compound, cone.Shape())
                n_built += 1

            else:
                raise ValueError(f"Unknown surface type: {stype}")

        except Exception as exc:
            print(f"[build_trimmed] face {i} ({stype}) failed: {exc}", file=sys.stderr)
            n_failed += 1
            continue

    if n_built == 0:
        raise RuntimeError("All faces failed to build")

    shape = compound

    # --- 7. Export BREP ---
    write_brep(shape, out_brep)

    # --- 7b. Export STEP (redirect stdout to avoid OCC printf noise) ---
    import os as _os
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone

    sys.stdout.flush()
    saved_fd = _os.dup(1)
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    try:
        _os.dup2(devnull_fd, 1)
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        status = writer.Write(out_step)
    finally:
        _os.dup2(saved_fd, 1)
        _os.close(saved_fd)
        _os.close(devnull_fd)
    if status != IFSelect_RetDone:
        raise RuntimeError("STEP export failed")

    step_size = _os.path.getsize(out_step)

    return {
        "n_faces_built": n_built,
        "n_faces_failed": n_failed,
        "brep_path": out_brep,
        "step_path": out_step,
        "step_size": step_size,
    }


def main():
    try:
        req = json.loads(sys.stdin.read())
        op = req["op"]
        if op == "build":
            res = op_build(req)
        elif op == "fillet":
            res = op_fillet_or_chamfer(req, "fillet")
        elif op == "chamfer":
            res = op_fillet_or_chamfer(req, "chamfer")
        elif op == "export":
            res = op_export(req)
        elif op == "build_trimmed":
            res = op_build_trimmed(req)
        else:
            raise ValueError(f"Unknown op: {op}")
        sys.stdout.write(json.dumps({"ok": True, "result": res}))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    main()

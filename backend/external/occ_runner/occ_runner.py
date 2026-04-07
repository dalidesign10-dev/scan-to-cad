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

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
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
        else:
            raise ValueError(f"Unknown op: {op}")
        sys.stdout.write(json.dumps({"ok": True, "result": res}))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    main()

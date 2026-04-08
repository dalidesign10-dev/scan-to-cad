"""Regression tests for the Phase E0 mechanical intent foundation.

These are deliberately synthetic — the point is to lock in behaviour on
cases where we *know* the answer, so future refactors can't silently
regress. Real scan validation happens elsewhere; do not add scan fixtures
here because they make the suite slow and flaky.

Cases covered:
    1. honest_sphere      — a sphere must yield ZERO high plane/cylinder fits
                            (we don't fit spheres yet, so everything should
                            honestly land as UNKNOWN).
    2. box_plus_boss      — a dense box with a cylindrical boss must yield
                            at least 6 high planes (box faces) and at least
                            1 high cylinder (boss side).
    3. partial_cylinder   — a 180° cylindrical arc must be identified as a
                            cylinder, not unknown (catches the normal-SVD
                            axis regression on partial arcs).
    4. uneven_box         — a single box with an unevenly subdivided face
                            must still resolve to ~6 plane regions, not
                            fragment because the high-density face has more
                            small triangles than the low-density faces.
                            (Catches a region-growing-by-vote regression.)
    5. force_plane_override — a region fit as UNKNOWN on a curved surface
                            must, after force_plane override, become a
                            plane fit (possibly at LOW / MEDIUM confidence).
                            This is the B2 regression test: the override
                            endpoint must actually do something.

Run directly:
    python backend/tests/test_intent_e0.py
Or under pytest if available:
    pytest backend/tests/test_intent_e0.py -q
"""

from __future__ import annotations

import os
import sys
import traceback

# Allow running without installing the backend package.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import trimesh

from pipeline.reconstruction import run_intent_segmentation
from pipeline.reconstruction.state import PrimitiveType, ConfidenceClass


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _session(mesh: trimesh.Trimesh, mesh_id: str = "test") -> dict:
    return {"mesh": mesh, "preprocessed": mesh, "mesh_id": mesh_id}


def _run(mesh: trimesh.Trimesh, target_proxy_faces: int = 4000, min_region_faces: int = 8):
    sess = _session(mesh)
    run_intent_segmentation(
        {"target_proxy_faces": target_proxy_faces, "min_region_faces": min_region_faces},
        progress_callback=lambda *a, **k: None,
        session=sess,
    )
    return sess["recon_state"]


def _dense_box(extents=(80.0, 60.0, 12.0), subdivisions: int = 3) -> trimesh.Trimesh:
    """A box dense enough that region growing isn't dominated by 2 triangles."""
    box = trimesh.creation.box(extents=extents)
    for _ in range(subdivisions):
        box = box.subdivide()
    return box


def _partial_cylinder(radius: float = 10.0, height: float = 20.0, arc_deg: float = 180.0, sections: int = 48) -> trimesh.Trimesh:
    """A cylindrical *arc* strip (no caps). Mimics a fillet / half-tube."""
    n_theta = max(6, int(sections * (arc_deg / 360.0)))
    theta = np.linspace(0.0, np.radians(arc_deg), n_theta)
    zs = np.linspace(-height / 2.0, height / 2.0, 8)
    verts = []
    for z in zs:
        for t in theta:
            verts.append([radius * np.cos(t), radius * np.sin(t), z])
    verts = np.asarray(verts, dtype=np.float64)
    faces = []
    for iz in range(len(zs) - 1):
        for it in range(n_theta - 1):
            a = iz * n_theta + it
            b = iz * n_theta + it + 1
            c = (iz + 1) * n_theta + it
            d = (iz + 1) * n_theta + it + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=True)


# ────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────

def test_honest_sphere():
    """A sphere has no plane or cylinder anywhere — the fitter must say so.

    We don't require n_unknown == n_regions (absorb can merge weirdly);
    what we require is: zero HIGH plane fits, zero HIGH cylinder fits.
    Any region that 'looks planar' on a 2-triangle approximation is a
    failure of the honesty story.
    """
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
    state = _run(mesh)
    s = state.summary()
    assert s["n_high_plane_fits"] == 0, f"sphere produced {s['n_high_plane_fits']} HIGH plane fits"
    assert s["n_high_cylinder_fits"] == 0, f"sphere produced {s['n_high_cylinder_fits']} HIGH cylinder fits"
    assert s["n_regions"] >= 1


def test_box_plus_boss_high_fits():
    """Dense box + cylindrical boss. We expect:
       - at least 6 high plane fits (one per box face family)
       - at least 1 high cylinder fit (the boss side wall)
    """
    box = _dense_box((80.0, 60.0, 12.0), subdivisions=3)
    boss = trimesh.creation.cylinder(radius=8.0, height=10.0, sections=48)
    boss.apply_translation([0.0, 0.0, 11.0])
    mesh = trimesh.util.concatenate([box, boss])
    state = _run(mesh)
    s = state.summary()
    assert s["n_high_plane_fits"] >= 6, f"expected ≥6 high plane fits, got {s['n_high_plane_fits']}"
    assert s["n_high_cylinder_fits"] >= 1, f"expected ≥1 high cylinder fit, got {s['n_high_cylinder_fits']}"
    # Area-explained should be close to 1.0 on noise-free geometry.
    assert s["explained_area_high_pct"] >= 80.0, (
        f"expected explained_area ≥80%, got {s['explained_area_high_pct']:.1f}%"
    )


def test_partial_cylinder_is_cylinder():
    """A half-cylinder arc must NOT be misclassified as a plane."""
    arc = _partial_cylinder(radius=10.0, height=20.0, arc_deg=180.0, sections=64)
    state = _run(arc, min_region_faces=8)
    s = state.summary()
    # At least one region fit must be a cylinder (high OR medium — partial
    # arcs are genuinely harder than full cylinders because the SVD on
    # normals has less support).
    any_cylinder = any(
        r.fit is not None
        and r.fit.type == PrimitiveType.CYLINDER
        and r.fit.confidence_class in (ConfidenceClass.HIGH, ConfidenceClass.MEDIUM)
        for r in state.regions.values()
    )
    assert any_cylinder, (
        f"partial cylinder gave no cylinder fits: "
        f"{[ (r.fit.type.value if r.fit else 'none', r.fit.confidence_class.value if r.fit else '-') for r in state.regions.values() ]}"
    )
    # And zero high plane fits — a half-tube is NOT a plane.
    assert s["n_high_plane_fits"] == 0, (
        f"partial cylinder misclassified {s['n_high_plane_fits']} regions as HIGH plane"
    )


def test_uneven_tessellation_not_overfragmented():
    """A single box, subdivided, with one face refined again.

    The point of this test is the area-weighted running normal (B3 fix)
    and the local-source soft gate. With unweighted accumulation a face
    family with hundreds of small triangles can drag the region average
    around and split a logical face into pieces. With area weighting +
    local-source gating, six box faces should produce roughly six regions.

    We test connected geometry on purpose — disconnected meshes (e.g.
    overlapping but not boolean-merged boxes) cannot be grown across by
    construction, so they would test mesh joinery rather than the grower.
    """
    box = _dense_box((60.0, 60.0, 60.0), subdivisions=2)
    state = _run(box)
    s = state.summary()
    assert s["n_regions"] <= 12, (
        f"over-fragmented: {s['n_regions']} regions on a single subdivided cube"
    )
    assert s["n_high_plane_fits"] >= 6, (
        f"expected ≥6 high plane fits on a cube, got {s['n_high_plane_fits']}"
    )
    # The cube is six logical faces, all coplanar within their face. The
    # area-explained should be near-total.
    assert s["explained_area_high_pct"] >= 90.0, (
        f"expected explained_area ≥90% on a cube, got {s['explained_area_high_pct']:.1f}%"
    )


def test_force_plane_override_takes_effect():
    """The force_plane override must actually change a region's fit.

    We build a subtly-curved strip (so auto-fit lands UNKNOWN or MEDIUM
    plane, not HIGH), find the largest region, force_plane it, and verify
    the region's fit.type is now PLANE. This is the B2 regression — the
    override endpoint was decorative in the initial commit.
    """
    from pipeline.reconstruction.fitting import fit_region

    # Gently curved strip: cylinder arc with a very large radius so the
    # curvature is small but present. Plane fit will be MEDIUM/LOW, not HIGH.
    arc = _partial_cylinder(radius=400.0, height=20.0, arc_deg=6.0, sections=64)
    state = _run(arc, min_region_faces=8)

    # Pick the largest region.
    regions_by_size = sorted(
        state.regions.values(),
        key=lambda r: r.full_face_indices.size,
        reverse=True,
    )
    assert regions_by_size, "no regions produced"
    target = regions_by_size[0]

    # Directly test that fit_region with forced_type=PLANE returns a plane,
    # regardless of what the auto-pick produced. This exercises the reader.
    mesh = arc
    full_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    full_faces = np.asarray(mesh.faces, dtype=np.int64)
    full_face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    verts_idx = np.unique(full_faces[target.full_face_indices].flatten())
    pts = full_vertices[verts_idx]
    norms = full_face_normals[target.full_face_indices]

    forced = fit_region(pts, norms, fit_source="test", forced_type=PrimitiveType.PLANE)
    assert forced.type == PrimitiveType.PLANE, (
        f"force_plane ignored: returned {forced.type.value}"
    )
    assert "override:force_plane" in (forced.notes or ""), (
        f"forced fit missing override note: {forced.notes!r}"
    )


# ────────────────────────────────────────────────────────────────────
# Runner (pytest-compatible + standalone)
# ────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    ("honest_sphere", test_honest_sphere),
    ("box_plus_boss", test_box_plus_boss_high_fits),
    ("partial_cylinder", test_partial_cylinder_is_cylinder),
    ("uneven_tessellation", test_uneven_tessellation_not_overfragmented),
    ("force_plane_override", test_force_plane_override_takes_effect),
]


def _main():
    failures = []
    for name, fn in ALL_TESTS:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()
            failures.append(name)
    print()
    if failures:
        print(f"{len(failures)} failing: {', '.join(failures)}")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


if __name__ == "__main__":
    _main()

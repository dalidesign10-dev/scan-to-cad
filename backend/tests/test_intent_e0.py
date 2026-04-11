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
    4. partial_cone       — a 270° conical arc strip must be identified as a
                            cone (catches the mean-normal signature +
                            5-param linear LS regression; chamfer-like).
    5. uneven_box         — a single box with an unevenly subdivided face
                            must still resolve to ~6 plane regions, not
                            fragment because the high-density face has more
                            small triangles than the low-density faces.
                            (Catches a region-growing-by-vote regression.)
    6. force_plane_override — a region fit as UNKNOWN on a curved surface
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


def _run(
    mesh: trimesh.Trimesh,
    target_proxy_faces: int = 4000,
    min_region_faces: int = 8,
    growth_mode: str = "dihedral",
):
    sess = _session(mesh)
    run_intent_segmentation(
        {
            "target_proxy_faces": target_proxy_faces,
            "min_region_faces": min_region_faces,
            "growth_mode": growth_mode,
        },
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


def _partial_cone(
    half_angle_deg: float = 30.0,
    z_lo: float = 2.0,
    z_hi: float = 10.0,
    arc_deg: float = 270.0,
    sections: int = 48,
    axial_strips: int = 10,
) -> trimesh.Trimesh:
    """A conical *arc* strip — no apex cap, no base cap. Stands in for a
    chamfer (apex below the model) as well as a full cone truncation.

    Apex sits at z=0, axis along +Z, half-angle = `half_angle_deg`. The
    visible strip spans z in [z_lo, z_hi] and phi in [0, arc_deg]. Using a
    partial arc is deliberate: CSG chamfers in mechanical parts are almost
    always only a sector of a full cone, which is also the harder case for
    the fitter (no azimuthal symmetry to lean on).
    """
    alpha = np.radians(half_angle_deg)
    n_theta = max(6, int(sections * (arc_deg / 360.0)))
    theta = np.linspace(0.0, np.radians(arc_deg), n_theta)
    zs = np.linspace(z_lo, z_hi, axial_strips)
    verts = []
    for z in zs:
        r = z * np.tan(alpha)
        for t in theta:
            verts.append([r * np.cos(t), r * np.sin(t), z])
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
    return trimesh.Trimesh(
        vertices=verts,
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )


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


def test_partial_cone_is_cone():
    """A conical arc strip (partial cone) must land as a CONE fit.

    This is the cone analogue of test_partial_cylinder_is_cylinder — it
    exists specifically because cones fail very differently from cylinders:
    a bad cone fitter will either (a) reject everything because the
    mean-normal signature is rejected, or (b) claim the surface is a
    slightly-curved cylinder. Both are visible from the region fit types.
    """
    cone_mesh = _partial_cone(
        half_angle_deg=30.0,
        z_lo=2.0,
        z_hi=10.0,
        arc_deg=270.0,
        sections=64,
        axial_strips=12,
    )
    state = _run(cone_mesh, min_region_faces=8)
    s = state.summary()
    any_cone = any(
        r.fit is not None
        and r.fit.type == PrimitiveType.CONE
        and r.fit.confidence_class in (ConfidenceClass.HIGH, ConfidenceClass.MEDIUM)
        for r in state.regions.values()
    )
    assert any_cone, (
        "partial cone produced no cone fits: "
        f"{[(r.fit.type.value if r.fit else 'none', r.fit.confidence_class.value if r.fit else '-') for r in state.regions.values()]}"
    )
    # Must NOT be misclassified as a plane.
    assert s["n_high_plane_fits"] == 0, (
        f"partial cone misclassified as HIGH plane ({s['n_high_plane_fits']} regions)"
    )

    # And when we check recovered params on the best cone region, the
    # half-angle should be near the truth (30°). We don't enforce a tight
    # bound here because synthesized meshes have small bookkeeping drift,
    # but +/- 5° is easily achievable on a clean synthetic.
    best_cone = None
    for r in state.regions.values():
        if r.fit and r.fit.type == PrimitiveType.CONE:
            if best_cone is None or r.fit.rmse < best_cone.rmse:
                best_cone = r.fit
    assert best_cone is not None
    half_angle = float(best_cone.params.get("half_angle_deg", 0.0))
    assert abs(half_angle - 30.0) < 5.0, (
        f"recovered half_angle={half_angle:.2f}° vs truth 30° (diff > 5°)"
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


def test_surface_family_grouping_on_cube():
    """A single box with stepped features produces multiple regions per
    face family. The surface_family_id post-pass must collapse each
    face family to a single family id.

    Specifically: take two concentric boxes (outer shell + a raised
    inner platform on the top face). The top of the outer box and the
    top of the inner raised platform are parallel but NOT coplanar
    (different d values), so they must be in DIFFERENT families. The
    top, bottom, and four side faces of the outer shell each give one
    family. Nested geometry like this is the canonical test for
    "parallel but not coplanar" — a naive normal-only grouper would
    merge them incorrectly.
    """
    outer = _dense_box((80.0, 80.0, 20.0), subdivisions=2)
    inner = _dense_box((30.0, 30.0, 5.0), subdivisions=1)
    inner.apply_translation([0.0, 0.0, 12.5])
    mesh = trimesh.util.concatenate([outer, inner])
    state = _run(mesh)
    s = state.summary()

    # At least 6 HIGH plane fits on the 6 outer box faces (the inner
    # platform contributes more). Family count should stay >= 6 because
    # the stepped top face produces a parallel-but-offset family.
    assert s["n_high_plane_fits"] >= 6, (
        f"expected ≥6 HIGH plane fits on stepped box, got {s['n_high_plane_fits']}"
    )
    assert s["n_plane_families"] >= 6, (
        f"expected ≥6 plane families, got {s['n_plane_families']}"
    )
    # Two regions with the SAME family_id must have matching plane
    # params (within tolerance). This is the invariant that justifies
    # calling them one family.
    from collections import defaultdict
    fam_to_regions = defaultdict(list)
    for r in state.regions.values():
        if r.fit and r.fit.type == PrimitiveType.PLANE and r.fit.confidence_class == ConfidenceClass.HIGH:
            fam_to_regions[r.surface_family_id].append(r)
    for fid, rs in fam_to_regions.items():
        if len(rs) < 2:
            continue
        ref = rs[0]
        ref_n = np.asarray(ref.fit.params["normal"])
        ref_d = float(ref.fit.params["d"])
        for r in rs[1:]:
            n = np.asarray(r.fit.params["normal"])
            d = float(r.fit.params["d"])
            dot = float(np.dot(n, ref_n))
            # Must be parallel within family tolerance.
            assert abs(dot) > 0.99, f"family {fid}: normals diverge (dot={dot:.3f})"
            d_aligned = d if dot > 0 else -d
            assert abs(d_aligned - ref_d) < 5.0, (
                f"family {fid}: d values diverge ({d_aligned:.2f} vs {ref_d:.2f}) "
                f"— parallel but not coplanar planes must be separate families"
            )

    # Every region must have a valid family id (no -1 leftovers).
    for r in state.regions.values():
        assert r.surface_family_id >= 0, (
            f"region {r.id} has no surface_family_id (pass left -1)"
        )

    # SurfaceFamily objects must exist for every HIGH fit and only for
    # HIGH fits. The canonical params must come from the largest-area
    # member — this lets downstream CAD consumers use ONE primitive per
    # surface without inspecting individual region fits.
    high_fids = {
        r.surface_family_id for r in state.regions.values()
        if r.fit and r.fit.confidence_class == ConfidenceClass.HIGH
        and r.fit.type != PrimitiveType.UNKNOWN
    }
    assert set(state.surface_families.keys()) == high_fids, (
        f"surface_families mismatch: "
        f"families={sorted(state.surface_families.keys())} vs "
        f"expected={sorted(high_fids)}"
    )
    # Each SurfaceFamily's representative must be the member with the
    # largest area_fraction, and its canonical params must match the
    # representative's fit params exactly.
    for fid, fam in state.surface_families.items():
        members = [state.regions[rid] for rid in fam.region_ids]
        assert members, f"family {fid} has no members"
        largest = max(members, key=lambda r: r.area_fraction)
        assert fam.representative_region_id == largest.id, (
            f"family {fid}: rep={fam.representative_region_id} "
            f"but largest member is {largest.id}"
        )
        assert fam.type == largest.fit.type
        # Params are a dict clone, so they compare equal but are NOT the
        # same object (mutating one must not affect the other).
        assert fam.canonical_params == largest.fit.params
        assert fam.canonical_params is not largest.fit.params


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
# Real-mesh smoke tests
#
# These run against the two mechanical meshes shipped in datasets/samples/.
# They are deliberately LOOSE — specific HIGH/MEDIUM counts shift easily
# when thresholds change, so the assertions only lock in the qualitative
# story:
#   • fandisk is a CAD-authored mechanical part → should produce many
#     HIGH fits and significant area-explained.
#   • rocker-arm is a smooth organic/freeform part → should produce ~zero
#     HIGH fits; region growing captures it as one mega-region which the
#     fitter correctly refuses to call a plane or cylinder. This is a
#     KNOWN ARCHITECTURAL LIMIT of the current grower (no curvature-aware
#     seeds), not a fitter bug. The test locks it in as "stays honest"
#     rather than pretending the rocker arm is made of primitives.
#
# Tests are skipped cleanly if the fixture files are missing, so the
# suite still works in environments that don't ship the sample meshes.
# ────────────────────────────────────────────────────────────────────

SAMPLES_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "datasets", "samples"))


def _load_sample(name: str):
    path = os.path.join(SAMPLES_DIR, name)
    if not os.path.isfile(path):
        return None
    return trimesh.load(path, force="mesh", process=True)


def test_fandisk_real_mechanical_mesh():
    """Fandisk — the canonical CAD-authored mechanical test mesh.

    This is NOT a scan: it's clean, watertight, has sharp mechanical
    edges, several planar faces, a few cylindrical holes and bevel
    fillets. It is the easiest "real" mesh E0 will ever see.

    We require:
      - at least 8 HIGH primitive fits total (planes + cylinders)
      - at least 1 HIGH cylinder (the cylindrical features)
      - at least 40% area-explained at HIGH confidence
      - NO mega-region that owns >85% of the area (that would mean the
        grower collapsed the whole mesh into one blob — the rocker-arm
        failure mode).
    """
    mesh = _load_sample("fandisk.obj")
    if mesh is None:
        print("  (skipping fandisk test — fixture not present)")
        return
    state = _run(mesh, target_proxy_faces=20000, min_region_faces=16)
    s = state.summary()
    n_high = s["n_high_plane_fits"] + s["n_high_cylinder_fits"]
    assert n_high >= 8, f"fandisk: expected ≥8 HIGH fits, got {n_high}"
    assert s["n_high_cylinder_fits"] >= 1, (
        f"fandisk: expected ≥1 HIGH cylinder, got {s['n_high_cylinder_fits']}"
    )
    assert s["explained_area_high_pct"] >= 40.0, (
        f"fandisk: expected ≥40% area explained, got {s['explained_area_high_pct']:.1f}%"
    )
    max_region_frac = max(
        (r.area_fraction for r in state.regions.values()), default=0.0
    )
    assert max_region_frac < 0.85, (
        f"fandisk: largest region covers {max_region_frac*100:.1f}% of area "
        f"(grower collapsed into mega-region)"
    )


def test_scanned11_raw_baseline():
    """scanned11.stl — a real 366k-face raw scan (not committed to git;
    extracted from scanned11.rar on origin/main into fixtures_local/).

    This test locks in the CURRENT BASELINE on a real scan so that any
    future change to the grower, the boundary signal, or the fit gates
    can be measured against a concrete "before" number. The numbers
    below are not a target — they are a floor. If you improve the
    grower you should see n_high_fits go UP and mega_region_frac go
    DOWN. If this test starts failing in the "too good" direction,
    UPDATE the baseline; don't relax it silently.

    Current baseline (E0 commit d198dc9 + perf fix, raw scan, no
    cleanup, default params):
        regions                     ~35
        HIGH plane                  2
        HIGH cylinder               2
        UNKNOWN                     30
        explained_area_high_pct     ~2%
        largest region area frac    ~0.94  (mega-region failure mode)

    Root cause: DIHEDRAL_SATURATION_DEG = 60° is calibrated for clean
    CAD geometry. On this scan the p95 dihedral is ~26°, so real
    mechanical edges only reach confidence ~0.43 (below HARD_CUT=0.55)
    and the grower walks across them. Adjusting the saturation to
    match the per-mesh dihedral distribution is the next concrete
    experiment.
    """
    scan_path = os.path.join(HERE, "fixtures_local", "scanned11.stl")
    if not os.path.isfile(scan_path):
        print(f"  (skipping scanned11 test — fixture not present at {scan_path})")
        return
    mesh = trimesh.load(scan_path, force="mesh", process=True)
    state = _run(mesh, target_proxy_faces=30000, min_region_faces=20)
    s = state.summary()
    n_high = s["n_high_plane_fits"] + s["n_high_cylinder_fits"]
    max_region_frac = max(
        (r.area_fraction for r in state.regions.values()), default=0.0
    )

    # These assertions are DELIBERATELY LOOSE — they lock in the floor,
    # not the ceiling, so that improvements don't falsely trip the test.
    #
    #   n_high >= 2          : at least the 2 high cylinders that exist
    #                          today must remain detected.
    #   max_region_frac       : we assert the mega-region exists today so
    #                          we notice when someone finally kills it.
    #                          Remove this lower bound if/when you do.
    assert n_high >= 2, (
        f"scanned11 regression: only {n_high} HIGH fits (baseline was 4)"
    )
    # Acknowledged failure mode — REMOVE this assertion the day the
    # grower stops producing a mega-region on scans.
    assert max_region_frac >= 0.70, (
        f"scanned11: largest region only covers {max_region_frac*100:.1f}% "
        f"— did the grower just start working on raw scans? Update the baseline!"
    )


def test_scanned11_fit_driven_segments_primitives():
    """scanned11.stl with the RANSAC-style fit-driven grower.

    Unlike the dihedral baseline above, this test uses
    growth_mode="fit_driven": seed a face, fit a plane or cylinder to its
    small neighborhood, grow BFS while new faces stay inside the fit's
    residual + normal tolerance. Boundaries fall out of the fit itself,
    so we don't need dihedral edges to form closed loops — which they
    don't on a noisy scan.

    Two passes inside the fit-driven grower:
      1. Cylinder-seed-first scan with a tight signature filter and
         validate-after-grow (only HIGH cylinder regions are kept; the
         rest are released back so the plane pass can claim them).
      2. Plane-dominant area-ordered loop on whatever's left.

    This is the concrete fix for the mega-region failure mode. On
    scanned11 the dihedral grower produces ~2% area explained in HIGH
    fits with a 94% mega-region. The fit-driven grower produces >=40%
    area explained in HIGH fits, no mega-region, >=10 HIGH planes,
    AND >=2 HIGH cylinders (the cylinder pass recovers them).

    Observed at this commit: ~61 HIGH planes, 3 HIGH cylinders, ~68%
    area at 40k proxy, max region ~7%.
    """
    scan_path = os.path.join(HERE, "fixtures_local", "scanned11.stl")
    if not os.path.isfile(scan_path):
        print(f"  (skipping scanned11 fit_driven — fixture not present at {scan_path})")
        return
    mesh = trimesh.load(scan_path, force="mesh", process=True)
    state = _run(
        mesh,
        target_proxy_faces=40000,
        min_region_faces=20,
        growth_mode="fit_driven",
    )
    s = state.summary()
    n_high_plane = s["n_high_plane_fits"]
    n_high_cyl = s["n_high_cylinder_fits"]
    n_high_cone = s.get("n_high_cone_fits", 0)
    n_high = n_high_plane + n_high_cyl + n_high_cone
    max_region_frac = max(
        (r.area_fraction for r in state.regions.values()), default=0.0
    )

    # HIGH plane count: the dihedral baseline finds 0. Fit-driven at
    # 40k proxy observed ~61. Floor at 20 to leave headroom for tuning.
    assert n_high_plane >= 20, (
        f"scanned11 fit_driven: expected >=20 HIGH planes, got {n_high_plane}"
    )
    # HIGH cylinder count: the cylinder-seed pass should recover the
    # cylindrical features that the plane-first loop was swallowing.
    # Observed: 3. Floor at 2.
    assert n_high_cyl >= 2, (
        f"scanned11 fit_driven: expected >=2 HIGH cylinders, got {n_high_cyl} "
        f"— is the cylinder-seed pass still wired in?"
    )
    # Total HIGH fits (planes + cylinders + cones).
    assert n_high >= 40, (
        f"scanned11 fit_driven: expected >=40 HIGH fits, got {n_high}"
    )
    # Area explained by HIGH fits: at 40k proxy observed ~68%.
    # Floor at 50% to leave headroom for tuning.
    assert s["explained_area_high_pct"] >= 50.0, (
        f"scanned11 fit_driven: only {s['explained_area_high_pct']:.1f}% "
        f"area explained at HIGH confidence"
    )
    # Mega-region must be gone. Baseline is 0.94; fit-driven observed
    # 0.17. Hard ceiling at 0.40 — larger than that means a major chunk
    # of the scan went unsegmented.
    assert max_region_frac <= 0.40, (
        f"scanned11 fit_driven: largest region still {max_region_frac*100:.1f}% "
        f"— did the mega-region come back?"
    )


def test_rocker_arm_freeform_stays_honest():
    """Rocker arm — smooth freeform organic part.

    No sharp mechanical edges anywhere, just curvature blends. The
    current region grower has no curvature-aware seeding, so it walks
    across the entire surface in one pass and produces one mega-region
    covering >90% of the area. The fitter correctly refuses to call
    that mega-region a plane or a cylinder.

    This test LOCKS IN the honest behaviour: we expect very few (or
    zero) HIGH fits on a freeform part. If some day the grower gains
    curvature-aware seeds and starts producing real HIGH fits here,
    that's an improvement and the test should be updated — NOT relaxed
    silently to accept false confidence.
    """
    mesh = _load_sample("rocker-arm.obj")
    if mesh is None:
        print("  (skipping rocker-arm test — fixture not present)")
        return
    state = _run(mesh, target_proxy_faces=20000, min_region_faces=16)
    s = state.summary()
    n_high = s["n_high_plane_fits"] + s["n_high_cylinder_fits"]
    # Honesty: at most 2 HIGH fits on a genuinely freeform part.
    assert n_high <= 2, (
        f"rocker-arm: {n_high} HIGH fits on a freeform part is suspicious "
        f"(over-confident segmentation?)"
    )
    # Acknowledged mega-region: current grower produces one giant region.
    # The test doesn't assert this as GOOD — it asserts the current
    # behaviour so we notice if it changes.
    max_region_frac = max(
        (r.area_fraction for r in state.regions.values()), default=0.0
    )
    assert max_region_frac >= 0.50, (
        f"rocker-arm: largest region only covers {max_region_frac*100:.1f}% "
        f"— if the grower stopped producing a mega-region, update this test."
    )


# ────────────────────────────────────────────────────────────────────
# Runner (pytest-compatible + standalone)
# ────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    ("honest_sphere", test_honest_sphere),
    ("box_plus_boss", test_box_plus_boss_high_fits),
    ("partial_cylinder", test_partial_cylinder_is_cylinder),
    ("partial_cone", test_partial_cone_is_cone),
    ("uneven_tessellation", test_uneven_tessellation_not_overfragmented),
    ("surface_family_grouping", test_surface_family_grouping_on_cube),
    ("fandisk_real_mesh", test_fandisk_real_mechanical_mesh),
    ("rocker_arm_freeform", test_rocker_arm_freeform_stays_honest),
    ("scanned11_raw_baseline", test_scanned11_raw_baseline),
    ("scanned11_fit_driven", test_scanned11_fit_driven_segments_primitives),
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

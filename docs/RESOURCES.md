# Scan-to-CAD Resources & Roadmap

Curated list of datasets, baselines, and libraries for building our reverse-engineering pipeline. Split into three practical buckets: paired scan↔CAD, synthetic CAD with surface ground truth, and feature-labeled datasets.

## Recommended shortlist

1. **ABC Dataset** — surface/primitive learning and benchmarks
2. **Fusion 360 Gallery (Segmentation + Reconstruction)** — fillet/chamfer labels and CAD-sequence supervision
3. **MFCAD + FeatureNet** — machining feature recognition
4. **Point2CAD** — baseline architecture to study first
5. **CGAL + Open3D + pythonOCC** — core implementation stack

## Datasets

### Paired Scan↔CAD

- **Scan2CAD** — classic paired scan/CAD when you need noisy scans aligned to CAD. Uses ScanNet scans + ShapeNet CAD models: 1,506 scans, 14,225 scan/CAD object pairs, 97,607 keypoint correspondences. Good for retrieval/alignment/pose, but targets indoor scene objects in RGB-D scans, not precise mechanical RE.

### Synthetic CAD with Ground-Truth Surfaces

- **ABC Dataset** — 1M CAD models with explicitly parameterized curves and surfaces. Provides ground truth for patch segmentation, geometric feature detection, and shape reconstruction. Includes a normal estimation benchmark with ground-truth normals from the parametric B-rep. **Best foundation for learning planes/cylinders/cones/curvature and patch labels.**

- **DeepCAD** — parsed from public Onshape documents linked from ABC. Includes tools to export to STEP. Treat as a sequence/program-style CAD resource, not first-line for raw scan surface fitting.

- **CAD-Recode** — newer; translates point clouds into executable Python CAD scripts. Introduces a procedurally generated dataset of 1M CAD programs. More ambitious and code-oriented; signals where the field is heading.

### Feature-Labeled Mechanical/CAD Datasets

- **Fusion 360 Gallery Dataset** — most useful public dataset if you care about engineering *features* rather than just geometry. Two subsets:
  - **Reconstruction Dataset**: 8,625 sequences
  - **Segmentation Dataset**: 35,680 parts, with faces labeled by the modeling operation that created them (Extrude, Fillet, Chamfer, etc.) — **highly relevant for teaching a system to recognize fillets/chamfers**.

- **BRepNet / Fusion segmentation benchmark** — topology-aware learning directly on CAD solids. Uses a benchmark dataset generated from Fusion 360 designs, built around B-rep topology (faces, edges, coedges) rather than just points/meshes.

- **MFCAD** — 3D CAD models labeled with machining features, provided in STEP format with helper scripts based on pythonocc-core. Good for detecting pockets, slots, holes, and machining-style structures.

- **FeatureNet / Machining-feature-dataset** — 24 classes × 1,000 models each, synthetic machining features. Not a full scan-to-CAD benchmark, but useful for bootstrapping a feature classifier.

## Baselines to Study

- **Point2CAD** ⭐ closest to our pipeline — takes a point cloud, segments into CAD-face clusters, fits each with a primitive or parametric surface, then **extends and intersects the surfaces so that topology emerges**. This is exactly the direction we're taking (Detect Features → infinite surfaces → intersection → B-Rep).

- **DeepCAD** — reconstructs CAD construction sequences.

- **CAD-Recode** — point cloud → editable CAD Python scripts.

- **BRepNet** — topology-aware segmentation on B-rep solids.

## Libraries

- **CGAL** — Efficient RANSAC and Region Growing with built-in detection for planes, spheres, cylinders, cones, tori.
- **Open3D** — RANSAC plane segmentation, planar patch detection.
- **PCL** — region growing by normals/curvature, sample-consensus cylinder segmentation.
- **pythonOCC / Open CASCADE** — Python access to OCCT's CAD/B-rep stack. Critical for rebuilding CAD entities after surface detection.

## Practical Training Order

For reverse engineering scanned mechanical parts with fillets/chamfers, public data is stronger on synthetic/procedural CAD than on true real-world metrology scan↔editable CAD pairs. Recommended order:

1. **ABC** for surface truth
2. **Fusion 360 Segmentation** for fillet/chamfer labels
3. **MFCAD / FeatureNet** for manufacturing features
4. **Real scans** only for fine-tuning and robustness testing

## Alignment with Our Current Pipeline

| Our stage | Resource to integrate |
|---|---|
| Segmentation | ABC patch labels + CGAL Efficient RANSAC as reference |
| Primitive fitting | ABC ground-truth surfaces for validation |
| Fillet/chamfer detection | Fusion 360 Gallery Segmentation labels — **train a classifier** on this |
| Feature detection (infinite surfaces) | Point2CAD architecture — study their intersect/extend logic |
| B-Rep reconstruction | pythonOCC (already in our stack) + Point2CAD topology emergence |
| STEP export | Already using OCCT |

## Next Steps (post-MVP)

1. Implement **surface-surface intersection** to generate edge curves from our infinite surfaces (Point2CAD-style)
2. Download a small ABC subset for validation of current primitive fitting
3. Train a simple Fusion 360 Segmentation classifier on fillet/chamfer face labels and use it to augment our curvature-based detection
4. Add MFCAD-style feature recognition as a second-pass post-process

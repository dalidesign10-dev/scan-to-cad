# Curated GitHub Repos for Scan-to-CAD Pipeline

A filtered list of repos worth saving for a real scan → surfaces/features → CAD/B-rep pipeline.

## Must-have now

- **prs-eth/point2cad** — reverse-engineers CAD from raw point clouds, reconstructing surfaces, edges, and corners. Closest to our goal — study this architecture first.
- **saali14/Scan-to-BRep** — BRepDetNet code for boundary/junction detection from 3D scans (CC3D + ABC). Best fit for scan-to-parametric-CAD logic.
- **AutodeskAILab/Fusion360GalleryDataset** — reconstruction, segmentation, assembly subsets. Segmentation labels include Fillet and Chamfer — critical for our use case.
- **deep-geometry/abc-dataset** — 1M CAD models with ground-truth parametric surfaces, patches, normals. Best validation target.
- **Open-Cascade-SAS/OCCT** + **tpaviot/pythonocc-core** — practical CAD reconstruction backbone for rebuilding editable geometry, fillets, trims, STEP output, solid topology. **Already in our stack.**
- **isl-org/Open3D** — preprocessing, normals, plane segmentation, meshing, registration, visualization. **Already in our stack.**

## End-to-end / closest to reverse engineering

- **prs-eth/point2cad** — point cloud → CAD faces/topology.
- **filaPro/cad-recode** — point clouds → Python CAD code. Reports on DeepCAD, Fusion360, CC3D benchmarks. Interesting for "editable CAD script" output later.
- **rundiwu/DeepCAD** — sequence/program-style CAD generation with STEP export and `pc2cad.py`.
- **skanti/Scan2CAD** — classic aligned scan↔CAD benchmark; more about alignment than RE, but useful for paired experiments.
- **saali14/Scan-to-BRep** — boundary/junction detection as the bridge from scan to CAD.

## Datasets and labels

- **AutodeskAILab/Fusion360GalleryDataset** — segmentation explicitly includes modeling operations: Extrude, Fillet, Chamfer.
- **deep-geometry/abc-dataset** — large CAD dataset for geometric deep learning.
- **hducg/MFCAD** — STEP models with machining feature labels.
- **madlabub/Machining-feature-dataset** — FeatureNet: 24 machining feature classes × 1,000 STL models each.
- **skanti/Scan2CAD** — paired scan/CAD annotations with correspondences and keypoints.

## B-rep / CAD-native learning

- **AutodeskAILab/BRepNet** — topological message passing on B-reps; face segmentation with topology awareness.
- **AutodeskAILab/UV-Net** — learns directly from B-reps via UV-grids on faces/edges; supports segmentation with MFCAD and Fusion 360 Gallery.
- **AutodeskAILab/JoinABLe** — assembly/joint learning on B-rep graphs. Useful later for multi-part reverse engineering.
- **AutodeskAILab/occwl** — lightweight Pythonic wrapper around pythonOCC; cleaner B-rep traversal.

## Geometry / point cloud / mesh stack

- **isl-org/Open3D** — preprocessing, normals, plane seg, meshing, registration, viz. **In use.**
- **isl-org/Open3D-ML** — 3D ML tooling on top of Open3D.
- **PointCloudLibrary/pcl** — classic C++ toolbox: segmentation, sample consensus, registration, features.
- **CGAL/cgal** — robust computational geometry, shape detection (Efficient RANSAC).
- **libigl/libigl** — mesh operators, curvature-related geometry processing.
- **leomariga/pyRANSAC-3D** — quick Python experiments for fitting planes/cuboids/cylinders/spheres.
- **pyvista/pyvista** — visualization/debugging for segmentation, fitted primitives, feature maps.

## CAD kernel / scripting / export

- **Open-Cascade-SAS/OCCT** — CAD kernel: solids, surfaces, topology, STEP/IGES, booleans, fillets/chamfers. **In use.**
- **tpaviot/pythonocc-core** — Python access to OCCT. **In use.**
- **CadQuery/cadquery** — parametric CAD scripting on top of OCCT; useful later for editable procedural CAD output.

## Priority for our app

### Must-have now
1. Point2CAD
2. Scan-to-BRep
3. Fusion360GalleryDataset
4. ABC dataset
5. Open3D *(in use)*
6. OCCT / pythonocc-core *(in use)*

### Add next
- BRepNet
- UV-Net
- MFCAD
- pyRANSAC-3D
- CGAL

### Watch for later
- CAD-Recode
- DeepCAD
- CadQuery
- JoinABLe

## Alignment with our current pipeline stages

| Pipeline stage | Repo to study/use |
|---|---|
| Mesh preprocessing | **Open3D** ✓ in use |
| Segmentation | **CGAL** (Efficient RANSAC reference), **BRepNet** for topology-aware learning |
| Primitive fitting | **pyRANSAC-3D** for Python patterns, **Point2CAD** architecture, **ABC** for validation |
| Feature detection (infinite surfaces) | **Point2CAD** ← direct match for our current step |
| Fillet/chamfer recognition | **Fusion360Gallery Segmentation**, **MFCAD**, **UV-Net** |
| B-Rep reconstruction | **pythonocc-core** ✓ in use, **Scan-to-BRep** (BRepDetNet), **occwl** helper |
| Multi-surface intersection & topology | **Point2CAD**, **OCCT** Boolean/trim APIs |
| STEP export | **pythonocc-core** ✓ in use |
| Editable CAD output (future) | **CadQuery**, **DeepCAD**, **CAD-Recode** |

## Immediate actions (recommended)

1. **Clone Point2CAD** and study its surface intersection / topology emergence code — this is the direct blueprint for our next step after the infinite surfaces we just built.
2. **Download ABC subset** (a few hundred models) to validate current plane/cylinder fitting accuracy against ground truth.
3. **Look at Fusion 360 Gallery Segmentation data** — face-level Fillet/Chamfer labels we could use to train a second-pass classifier to augment our curvature-based fillet band detection.
4. **Add occwl** as a convenience wrapper for future B-Rep traversal code.

## Ranked roadmap (study / clone / train order)

### Study first (read code / papers)
1. **Point2CAD** — architectural reference for our whole pipeline
2. **BRepNet** paper/repo — for understanding topology-aware face segmentation
3. **Scan-to-BRep** (BRepDetNet) — for the scan→parametric-CAD bridge

### Clone first (run locally, integrate techniques)
1. **Point2CAD** — run its inference path on our clio3.stl point cloud for comparison
2. **pyRANSAC-3D** — quick primitive fitting baseline to compare against ours
3. **occwl** — integrate for cleaner OCC B-Rep code

### Train first (if we decide to add ML)
1. A small classifier on **Fusion 360 Gallery Segmentation** for Fillet/Chamfer face labels
2. Validate segmentation quality against **ABC** ground truth
3. Only later: FeatureNet/MFCAD for manufacturing feature recognition

### Use only later
- CAD-Recode (editable scripts) — too ambitious for current stage
- DeepCAD (construction sequences) — post-MVP
- CadQuery (parametric output) — post-MVP
- JoinABLe (assemblies) — when we move beyond single-part

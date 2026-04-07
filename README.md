# Scan-to-CAD

Reverse-engineering app that turns 3D scans of mechanical parts into editable CAD geometry.

Built as an Electron + React frontend over a Python FastAPI backend, with classical
geometry-processing pipelines (cleanup, segmentation, primitive fitting) and OCC-based
B-Rep operations (chamfer / fillet / STEP export). Includes an experimental Point2Cyl
AI integration as a side-channel.

---

## Status

This is a working prototype, not a finished product. Honest list of what works and what doesn't is at the bottom.

### What works today
- **Phase A — Cleanup**: noisy scan → watertight Poisson-reconstructed mesh (≈30 s on a typical bracket).
- **Phase B — Cage extraction**: segments + primitive fits merged into "logical CAD faces" via union-find. Useful as input to manual workflows.
- **Phase C — Construction features**: decimate → polyhedral OCC solid → apply chamfer/fillet on sharp edges → export as STL / STEP / BREP.
- **Manual plane workflow**: click segments, merge them, fit infinite planes from selection, intersect plane pairs to recover edges, undo/redo with Ctrl+Z/Y.
- **Point2Cyl AI**: pretrained PointNet++ extrusion decomposition (works on simple parts; out-of-distribution on complex scans).

### What doesn't (yet)
- The polyhedral B-Rep is still scan-derived faceted geometry, not true analytic CAD.
- Cage extraction over-segments (~300 faces on a typical bracket) — needs VSA upgrade.
- No design-intent reconstruction yet (coaxial snapping, true-circle holes, recovered fillet bands).
- Old toolbar Export buttons go through a deprecated segmentation export path; use the Phase C card's export instead.

---

## Architecture

```
┌──────────────────────┐         ┌──────────────────────┐
│  Electron / React    │  HTTP   │  FastAPI backend     │
│  + three.js viewport │ ──────► │  (port 8321)         │
│  + zustand store     │         │                      │
└──────────────────────┘         │  trimesh, open3d,    │
                                 │  numpy, scipy        │
                                 └──────┬───────────────┘
                                        │ subprocess
                       ┌────────────────┴─────────────────┐
                       ▼                                  ▼
              ┌────────────────────┐            ┌──────────────────┐
              │ conda env: occ     │            │ conda env:       │
              │ pythonocc-core 7.8 │            │ point2cyl        │
              │ → BRep / STEP /    │            │ torch + scatter  │
              │   fillet / chamfer │            │ → AI inference   │
              └────────────────────┘            └──────────────────┘
```

OCC and Point2Cyl are isolated in dedicated conda envs because their native dependencies
conflict with the main backend. The main backend talks to them via subprocess + JSON.

### Key directories
- `backend/src/pipeline/` — geometry pipeline (cleanup, segmentation, primitive_fitting, cage_extraction, features, …)
- `backend/src/http_server.py` — FastAPI app exposing all stages
- `backend/external/occ_runner/` — OCC operations runner (uses `occ` conda env)
- `backend/external/point2cyl/` — Point2Cyl AI inference runner (uses `point2cyl` conda env)
- `frontend/src/renderer/` — React UI + three.js viewport + zustand store
- `frontend/src/renderer/components/PipelinePanel.tsx` — the pipeline step buttons
- `frontend/src/renderer/three/` — viewport overlays (segmentation, primitives, edges, polyhedral CAD, Point2Cyl)

---

## Setup

### Prerequisites
- Windows (tested on Win 11)
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Node.js ≥ 18
- An NVIDIA GPU is helpful for the Point2Cyl side-channel but not required for the main pipeline.

### 1. Main backend (Python 3.10)
```bash
cd backend
pip install -r requirements.txt
```
This pulls `numpy`, `scipy`, `trimesh`, `open3d`, `fastapi`, `uvicorn`. Note: `pythonocc-core` is **not** installed here — it lives in its own conda env (next step).

### 2. OCC subprocess env (for Phase C / fillet / chamfer / STEP export)
```bash
conda create -n occ -c conda-forge --override-channels python=3.10 pythonocc-core=7.8.1 numpy trimesh -y
```

### 3. Point2Cyl env (optional — only if you want the AI side-channel)
```bash
conda create -n point2cyl -c conda-forge --override-channels python=3.10 -y
conda run -n point2cyl pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
conda run -n point2cyl pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
conda run -n point2cyl pip install trimesh
```
The Point2Cyl repo and pretrained weights are already in `backend/external/point2cyl/`.

### 4. Frontend
```bash
cd frontend
npm install
```

### 5. Hardcoded conda paths
The subprocess wrappers in `backend/src/pipeline/features.py` and `http_server.py`
currently hardcode the conda env python paths to `C:\Users\Dali Design\miniconda3\envs\...`.
**Edit these to your own miniconda location** before running.

---

## Running

In two terminals:

```bash
# Terminal 1 — backend
cd backend
python src/http_server.py     # http://localhost:8321
```

```bash
# Terminal 2 — frontend
cd frontend
npx vite --config vite.config.ts   # http://localhost:5173
```

Open `http://localhost:5173` in a browser. The app talks to the backend on port 8321.

---

## Usage walkthrough

1. **Open Mesh** → load a scanned STL.
2. **Run Cleanup** (1b card) → Poisson-reconstructed watertight mesh.
3. *(Optional)* **Run Segmentation** + **Fit Primitives** → analytic patch fits.
4. *(Optional)* **Extract Cage** (4b card) → merge primitive patches into logical CAD faces.
5. **Build Polyhedral B-Rep** (Phase C card, target_faces ≈ 2000) → real OCC solid.
6. **Fillet** or **Chamfer** sharp edges (use threshold ≥ 100° on the polyhedral solid — lower thresholds make OCC bail on overlapping fillets).
7. **Export STL / STEP / BREP** from the Phase C card.

Manual workflow alongside this:
- Click any segment patch in the viewport to select it
- Shift-click for multi-select
- Merge selected, or create an infinite plane from selection
- Pick two planes from the chip list and click **Intersect Selected** to get the shared edge
- **Ctrl+Z / Ctrl+Y** to undo / redo

---

## What this repo includes

- `backend/src/` — main FastAPI server + geometry pipelines
- `backend/external/` — third-party tools (Point2Cyl repo + pretrained weights, OCC runner)
- `frontend/` — full Electron + React + three.js client
- `docs/` — internal notes
- `references/` — reference implementations / papers

Total ≈ 100 MB committed (most of it the Point2Cyl weights + repo).

---

## License

MIT for the original code in this repo.
Point2Cyl and other vendored third-party code retain their original licenses (see `backend/external/point2cyl/LICENSE`).

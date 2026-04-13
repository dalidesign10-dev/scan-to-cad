# Next Session: Edge-First AI Reconstruction

## Vision
Build CAD geometry like a real reverse engineer:
1. Find all sharp edges on the mesh
2. Trace edges into clean lines/arcs
3. AI classifies each surface between edges
4. AI creates perfect surfaces bounded by those edges
5. All happening live in the viewport

## What's Already Built
- E0: Region segmentation + primitive fitting (planes/cylinders/cones)
- E0 has sharp edge detection (boundary signals with dihedral angle + confidence)
- E1: Vertex snapping to analytic surfaces
- AI classifier using Claude API (per-region classification)
- Live reconstruction SSE streaming (region-by-region viewport updates)
- Deviation analysis (mesh-to-CAD comparison)
- Progress polling for E0 (detailed step-by-step updates)

## What Needs Building

### 1. Edge Detection + Tracing
- Use E0's `proxy_edge_confidence` and `proxy_edge_endpoints` (already computed)
- Chain sharp edges into continuous edge curves
- Simplify each edge curve (Douglas-Peucker → clean straight lines or arcs)
- Classify edges: straight line, arc, freeform

### 2. Edge-First Surface Construction  
- For each surface bounded by edges:
  - AI looks at geometric features → decides surface type
  - System creates the surface (plane, cylinder, cone)
  - Boundary = the traced edges around this surface
  - Result: clean polygon/patch bounded by clean edges

### 3. Live Viewport Rendering
- Show edges appearing as clean lines (color-coded: straight=blue, arc=green)
- Show surfaces filling in between edges
- Show deviation overlay per surface
- All in real-time via SSE streaming

## Key Files
- `backend/src/pipeline/reconstruction/pipeline.py` — E0 orchestration
- `backend/src/pipeline/reconstruction/boundary.py` — edge detection
- `backend/src/pipeline/reconstruction/reconstruct_live.py` — SSE streaming
- `backend/src/pipeline/reconstruction/cad_export.py` — geometry construction
- `backend/src/pipeline/reconstruction/ai_classify.py` — Claude API classifier
- `frontend/src/renderer/three/SceneManager.ts` — viewport rendering

## API Key
ANTHROPIC_API_KEY needs to be set as env var before launching the app.

## How to Run
```bash
cd D:/ClaudeTEST/scan-to-cad/frontend
set ANTHROPIC_API_KEY=<key>
npm run dev
```

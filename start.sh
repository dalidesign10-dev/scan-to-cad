#!/usr/bin/env bash
# start.sh — one-command launcher for the scan-to-CAD desktop app.
#
# Launches the full Electron app: one window, one process tree.
# Electron's main process spawns the Python FastAPI backend as a child
# and kills it when you close the window.
#
# Phase C (OCC fillet/chamfer/STEP export) and Point2Cyl still require
# the separate conda envs described in README.md — those run via
# subprocess only when you click the corresponding buttons. Phase E0
# (Intent Reconstruction) and the main mesh pipeline work with just
# the main backend requirements.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[start.sh] error: no python3 or python on PATH" >&2
    exit 1
  fi
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "[start.sh] error: npx not found — install Node.js 18+ first" >&2
  exit 1
fi

# First-run convenience: install frontend deps if missing.
if [ ! -d "frontend/node_modules" ]; then
  echo "[start.sh] frontend/node_modules missing — running 'npm install' (one-time)..."
  (cd frontend && npm install) || {
    echo "[start.sh] npm install failed" >&2
    exit 1
  }
fi

# Quick import check so we fail fast with a clear message if the Python
# deps aren't installed. We import the heavy deps the E0 path actually
# uses.
if ! "$PYTHON_BIN" -c "import fastapi, uvicorn, trimesh, numpy, scipy" 2>/dev/null; then
  echo "[start.sh] error: backend Python deps missing." >&2
  echo "           run:  $PYTHON_BIN -m pip install -r backend/requirements.txt" >&2
  exit 1
fi

echo "[start.sh] launching Electron desktop app..."
echo "           Electron will spawn the Python backend as a child"
echo "           process and open the app window. Closing the window"
echo "           exits both."
echo

cd frontend
exec npm run dev

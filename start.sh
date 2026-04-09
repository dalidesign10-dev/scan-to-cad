#!/usr/bin/env bash
# start.sh — one-command launcher for the scan-to-CAD app.
#
# Starts the FastAPI backend on http://localhost:8321 and the Vite dev
# server on http://localhost:5173, then opens the frontend URL so you can
# drive the pipeline from a browser. Ctrl+C kills both.
#
# Phase C (OCC fillet/chamfer/STEP export) and Point2Cyl still require
# the separate conda envs described in README.md — those run via
# subprocess only when you click the corresponding buttons. Phase E0
# (Intent Reconstruction) and the main mesh pipeline work with just the
# main backend requirements.
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

BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  # Kill both children on any exit. || true because they may already be dead.
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[start.sh] starting backend on http://localhost:8321"
(
  cd backend
  exec "$PYTHON_BIN" src/http_server.py
) &
BACKEND_PID=$!

# Wait up to 15s for the backend to become reachable. We probe /docs
# (FastAPI auto-generates it) rather than /api/... to avoid pipeline work.
for i in $(seq 1 30); do
  if curl -sf http://localhost:8321/docs >/dev/null 2>&1; then
    echo "[start.sh] backend is up"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "[start.sh] backend exited before it became ready — see logs above" >&2
    exit 1
  fi
  sleep 0.5
done

echo "[start.sh] starting frontend on http://localhost:5173"
(
  cd frontend
  exec npx vite --config vite.config.ts
) &
FRONTEND_PID=$!

# Try to open the browser once the frontend likely has a listening
# socket. Silent if no opener is available — the URL is printed anyway.
(
  for i in $(seq 1 30); do
    if curl -sf http://localhost:5173 >/dev/null 2>&1; then
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open http://localhost:5173 >/dev/null 2>&1 || true
      elif command -v open >/dev/null 2>&1; then
        open http://localhost:5173 >/dev/null 2>&1 || true
      fi
      break
    fi
    sleep 0.5
  done
) &

echo
echo "  Backend:  http://localhost:8321  (FastAPI, /docs for API)"
echo "  Frontend: http://localhost:5173  (open in browser)"
echo
echo "  Phase E0: load a mesh -> Cleanup -> Preprocess -> E0. Intent"
echo "            Reconstruction card -> flip 'Growth mode' to"
echo "            'fit_driven (scans)' -> Run E0 Intent."
echo
echo "  Ctrl+C to stop both."
echo

# Wait on either child; if one dies, fall through and cleanup() kills
# the other.
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true

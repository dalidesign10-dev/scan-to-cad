@echo off
REM start.bat — one-command launcher for the scan-to-CAD desktop app.
REM
REM Double-click this file (or run from a cmd/PowerShell prompt). It
REM launches the full Electron desktop app: one window, one process
REM tree. Electron's main process spawns the Python FastAPI backend as
REM a child and kills it when the window closes.
REM
REM Phase C (OCC fillet/chamfer/STEP export) and Point2Cyl still need
REM the separate conda envs described in README.md — those run via
REM subprocess only when you click the corresponding buttons. Phase E0
REM (Intent Reconstruction) and the main mesh pipeline work with just
REM the main backend requirements.

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

where npx >nul 2>&1
if errorlevel 1 (
  echo [start.bat] error: npx not found. Install Node.js 18+ first.
  pause
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo [start.bat] error: python not on PATH. Install Python 3.10+ first.
  pause
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [start.bat] frontend\node_modules missing - running npm install (one-time)...
  pushd frontend
  call npm install
  if errorlevel 1 (
    echo [start.bat] npm install failed
    popd
    pause
    exit /b 1
  )
  popd
)

python -c "import fastapi, uvicorn, trimesh, numpy, scipy" >nul 2>&1
if errorlevel 1 (
  echo [start.bat] error: backend Python deps missing.
  echo            run:  python -m pip install -r backend\requirements.txt
  pause
  exit /b 1
)

echo [start.bat] launching Electron desktop app...
echo             Electron will spawn the Python backend automatically
echo             and open the app window. Closing the window exits.
echo.

cd /d "%SCRIPT_DIR%frontend"
call npm run dev

REM If npm run dev exits non-zero, pause so the user can read the error.
if errorlevel 1 (
  echo.
  echo [start.bat] npm run dev exited with error code %errorlevel%
  pause
)

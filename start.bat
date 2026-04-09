@echo off
REM start.bat — one-command launcher for the scan-to-CAD app (Windows).
REM
REM Starts the FastAPI backend on http://localhost:8321 in a separate
REM window, then starts the Vite dev server on http://localhost:5173
REM in this window. Close either window to stop that half.
REM
REM Phase C (OCC) and Point2Cyl still need the conda envs described in
REM README.md — those run via subprocess only when you click the
REM corresponding buttons. Phase E0 works with just the main backend
REM requirements.

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

where npx >nul 2>&1
if errorlevel 1 (
  echo [start.bat] error: npx not found — install Node.js 18+ first
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo [start.bat] error: python not on PATH
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [start.bat] frontend\node_modules missing — running npm install (one-time)...
  pushd frontend
  call npm install
  if errorlevel 1 (
    echo [start.bat] npm install failed
    popd
    exit /b 1
  )
  popd
)

python -c "import fastapi, uvicorn, trimesh, numpy, scipy" >nul 2>&1
if errorlevel 1 (
  echo [start.bat] error: backend Python deps missing.
  echo            run:  python -m pip install -r backend\requirements.txt
  exit /b 1
)

echo [start.bat] starting backend on http://localhost:8321 (new window)
start "scan-to-cad backend" cmd /k "cd /d %SCRIPT_DIR%backend && python src\http_server.py"

REM Give the backend a moment to bind the port.
timeout /t 3 /nobreak >nul

echo [start.bat] starting frontend on http://localhost:5173
echo.
echo   Backend:  http://localhost:8321  (FastAPI, /docs for API)
echo   Frontend: http://localhost:5173  (opens automatically)
echo.
echo   Phase E0: load a mesh -^> Cleanup -^> Preprocess -^> E0. Intent
echo             Reconstruction card -^> flip 'Growth mode' to
echo             'fit_driven (scans)' -^> Run E0 Intent.
echo.
echo   Close this window to stop the frontend; close the backend
echo   window separately to stop the backend.
echo.

start "" http://localhost:5173

cd /d "%SCRIPT_DIR%frontend"
npx vite --config vite.config.ts

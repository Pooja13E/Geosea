@echo off
setlocal
cd /d "%~dp0"
echo.
echo =====================================
echo GeoSea - Start Dashboard and API
echo =====================================
echo.
if not exist package.json (
  echo ERROR: package.json was not found in this folder.
  echo Extract the ZIP and run this file from the GeoSea_CLEAN_FINAL folder.
  pause
  exit /b 1
)
if not exist backend\app.py (
  echo ERROR: backend\app.py was not found. The project folder is incomplete.
  pause
  exit /b 1
)
if not exist node_modules (
  echo Installing frontend dependencies for this GeoSea copy...
  call npm install
  if errorlevel 1 (
    echo npm install failed. Check Node.js/npm and your internet connection.
    pause
    exit /b 1
  )
)
echo Starting GeoSea API in a separate terminal...
start "GeoSea Backend API" /D "%~dp0backend" cmd /k "py -m uvicorn app:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
echo Starting GeoSea dashboard...
echo Backend health check: http://127.0.0.1:8000/health
echo Frontend: http://localhost:5173
echo Keep both terminal windows open while using GeoSea.
call npm run dev
pause

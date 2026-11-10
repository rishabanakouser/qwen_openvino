@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"

"%PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000
if errorlevel 1 (
    echo.
    echo Failed to start the service. Check that Python and the project dependencies are installed.
    pause
    exit /b 1
)

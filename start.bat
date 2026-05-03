@echo off
title LLM Adapter - Startup
color 0A

echo ============================================
echo        LLM ADAPTER - Starting All Services
echo ============================================
echo.

:: ─── 1. Start Ollama ────────────────────────────
echo [1/4] Starting Ollama server...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if %ERRORLEVEL%==0 (
    echo       Ollama already running. Skipping.
) else (
    start "Ollama Server" cmd /c "ollama serve"
    timeout /t 3 /nobreak >NUL
    echo       Ollama started.
)
echo.

:: ─── 2. Pull models if needed ───────────────────
echo [2/4] Ensuring models are available...
ollama list | find "tinyllama" >NUL 2>&1 || (echo       Pulling tinyllama... & ollama pull tinyllama)
ollama list | find "phi3" >NUL 2>&1 || (echo       Pulling phi3... & ollama pull phi3)
ollama list | find "mistral" >NUL 2>&1 || (echo       Pulling mistral... & ollama pull mistral)
echo       All models ready.
echo.

:: ─── 3. Start Prometheus & Grafana (Docker) ─────
echo [3/4] Starting Prometheus + Grafana...
where docker >NUL 2>&1
if %ERRORLEVEL%==0 (
    docker compose up -d prometheus grafana 2>NUL
    if %ERRORLEVEL%==0 (
        echo       Prometheus: http://localhost:9090
        echo       Grafana:    http://localhost:3000  (admin/admin)
    ) else (
        echo       Docker Compose failed. Prometheus/Grafana skipped.
        echo       (Install Docker Desktop or run them manually)
    )
) else (
    echo       Docker not found. Prometheus/Grafana skipped.
    echo       (Install Docker Desktop to enable monitoring)
)
echo.

:: ─── 4. Start FastAPI Backend + Frontend ────────
echo [4/4] Starting LLM Adapter backend...
echo.
echo ============================================
echo   All services starting!
echo.
echo   Dashboard:   http://localhost:8000
echo   API Docs:    http://localhost:8000/docs
echo   Prometheus:  http://localhost:9090
echo   Grafana:     http://localhost:3000
echo ============================================
echo.
echo   Press Ctrl+C to stop the server.
echo.

cd /d "%~dp0"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause

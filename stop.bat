@echo off
title LLM Adapter - Shutdown
color 0C

echo ============================================
echo        LLM ADAPTER - Stopping All Services
echo ============================================
echo.

echo [1/3] Stopping Prometheus + Grafana...
where docker >NUL 2>&1
if %ERRORLEVEL%==0 (
    docker compose down 2>NUL
    echo       Docker services stopped.
) else (
    echo       Docker not found. Skipped.
)
echo.

echo [2/3] Stopping Ollama...
taskkill /IM ollama.exe /F >NUL 2>&1
echo       Ollama stopped.
echo.

echo [3/3] Stopping any uvicorn processes...
taskkill /IM uvicorn.exe /F >NUL 2>&1
echo       Uvicorn stopped.
echo.

echo ============================================
echo   All services stopped.
echo ============================================

timeout /t 3

@echo off
setlocal
cd /d "%~dp0"

title Hyper Captioner

:: Force UTF-8 encoding for Unicode captions/tags
set PYTHONUTF8=1

:: Enforce 100% Offline Mode (No HuggingFace or telemetry requests)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set GRADIO_ANALYTICS_ENABLED=False

echo ===================================================
echo               HYPER CAPTIONER
echo ===================================================
echo.

:: Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call "venv\Scripts\activate.bat"
) else (
    echo [!] No venv found. Using system Python...
)

echo [*] Launching Hyper Captioner UI...
echo.

python run.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Application exited with an error (Code: %ERRORLEVEL%).
    pause
)

endlocal


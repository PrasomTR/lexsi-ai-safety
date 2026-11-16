@echo off
setlocal EnableDelayedExpansion

echo ======================================================================
echo   Lexsi AI Safety Hackathon (Track 2: AI Red Teaming ^& Safety)
echo   Environment Initialization and Hardware Verification Script
echo ======================================================================
echo.

:: 1. Verify Python availability
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.12 and add it to PATH.
    exit /b 1
)
python --version
echo [OK] Python is available.
echo.

:: 2. Upgrade pip and build tooling via python module (never run standalone pip.exe)
echo [2/6] Upgrading pip, setuptools, and wheel via python module...
python -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to upgrade build tooling.
    exit /b 1
)
echo [OK] Build tooling updated.
echo.

:: 3. Install PyTorch with CUDA support (CUDA 12.8 primary for Blackwell RTX 5060; fallback to 12.4)
echo [3/6] Installing PyTorch with CUDA support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %ERRORLEVEL% neq 0 (
    echo [WARNING] PyTorch cu128 install failed, falling back to cu124...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install PyTorch with CUDA support.
        exit /b 1
    )
)
echo [OK] PyTorch installed successfully.
echo.

:: 4. Install project dependencies from requirements.txt
echo [4/6] Installing project dependencies from requirements.txt...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install requirements.txt.
    exit /b 1
)
echo [OK] Dependencies installed successfully.
echo.

:: 5. Install editable package
echo [5/6] Installing local project in editable mode...
python -m pip install -e . --no-deps
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Editable install failed; continuing...
)
echo.

:: 6. Run Environment & Hardware Verification
echo [6/6] Verifying environment, GPU acceleration, and Track 2 toolkits...
python -c "import sys; print('Python executable:', sys.executable); print('Python version:', sys.version.split()[0]); import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); import transformers; print('Transformers:', transformers.__version__); import peft; print('PEFT:', peft.__version__); import bitsandbytes; print('BitsAndBytes:', bitsandbytes.__version__); import auditkit; print('AuditKIT:', auditkit.__version__); import safetune; print('SafeTune:', safetune.__version__); import curatorkit; print('CuratorKIT:', curatorkit.__version__);"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Environment verification failed.
    exit /b 1
)

echo.
echo ======================================================================
echo   [SUCCESS] Environment setup and verification completed successfully!
echo ======================================================================
exit /b 0

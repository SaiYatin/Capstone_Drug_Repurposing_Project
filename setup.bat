@echo off
echo ============================================================
echo BioPathNet++ Setup Script for RTX 4090
echo ============================================================
echo.

REM Check Python
echo [1/10] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Install Python 3.10 from python.org
    pause
    exit /b 1
)
echo OK
echo.

REM Check CUDA
echo [2/10] Checking CUDA...
nvidia-smi
if %errorlevel% neq 0 (
    echo ERROR: NVIDIA GPU not detected!
    pause
    exit /b 1
)
echo OK
echo.

REM Create virtual environment
echo [3/10] Creating virtual environment...
python -m venv venv
echo OK
echo.

REM Activate venv
echo [4/10] Activating virtual environment...
call venv\Scripts\activate.bat
echo OK
echo.

REM Upgrade pip
echo [5/10] Upgrading pip...
python -m pip install --upgrade pip
echo OK
echo.

REM Install PyTorch with CUDA
echo [6/10] Installing PyTorch with CUDA 12.1...
echo This may take 5-10 minutes...
echo (Compatible with CUDA 12.x and 13.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo OK
echo.

REM Install PyTorch Geometric
echo [7/10] Installing PyTorch Geometric...
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
echo OK
echo.

REM Install other dependencies
echo [8/10] Installing other dependencies...
pip install tensorboard scikit-learn tqdm pandas numpy
echo OK
echo.

REM Create directories
echo [9/10] Creating project directories...
if not exist "models" mkdir models
if not exist "training" mkdir training
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "datasets\processed" mkdir datasets\processed
echo OK
echo.

REM Test GPU
echo [10/10] Testing GPU configuration...
python test_gpu.py
echo.

echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. Copy model files to models/
echo   2. Copy train_optimized.py to training/
echo   3. Copy data files to datasets/processed/
echo   4. Run: cd training
echo   5. Run: python train_optimized.py
echo.
pause

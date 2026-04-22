@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV=.venv_stt"
set "PYEXE=%VENV%\Scripts\python.exe"

if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin" set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin" set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;%PATH%"
if exist "C:\Program Files\NVIDIA\CUDNN\v9.0\bin" set "PATH=C:\Program Files\NVIDIA\CUDNN\v9.0\bin;%PATH%"
if exist "C:\Program Files\NVIDIA\CUDNN\v9.1\bin" set "PATH=C:\Program Files\NVIDIA\CUDNN\v9.1\bin;%PATH%"

echo ========================================
echo  Whisper GPU Dependency Check
echo ========================================
echo.

where nvidia-smi
if errorlevel 1 echo [WARN] nvidia-smi was not found in PATH.
echo.

nvidia-smi
echo.

where cublas64_12.dll
if errorlevel 1 echo [MISSING] cublas64_12.dll

where cudnn64_9.dll
if errorlevel 1 echo [MISSING] cudnn64_9.dll

echo.
"%PYEXE%" -c "import ctranslate2; print('ctranslate2', ctranslate2.__version__)"
echo.
echo If cublas64_12.dll or cudnn64_9.dll is missing, install cuBLAS for CUDA 12 and cuDNN 9 for CUDA 12, then ensure their bin folders are in PATH.
pause

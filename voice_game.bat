@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV=.venv_stt"
set "PYEXE=%VENV%\Scripts\python.exe"
set "MODEL=medium"
set "EXTRA_ARGS="
set "HF_HOME=%CD%\.hf_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin" set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin" set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;%PATH%"
if exist "C:\Program Files\NVIDIA\CUDNN\v9.0\bin" set "PATH=C:\Program Files\NVIDIA\CUDNN\v9.0\bin;%PATH%"
if exist "C:\Program Files\NVIDIA\CUDNN\v9.1\bin" set "PATH=C:\Program Files\NVIDIA\CUDNN\v9.1\bin;%PATH%"

if not "%~1"=="" (
  set "FIRST=%~1"
  if not "%FIRST:~0,2%"=="--" (
    set "MODEL=%~1"
    shift
  )
)

:collect_args
if "%~1"=="" goto :args_done
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto :collect_args
:args_done

echo ========================================
echo  Voice Context Console
echo ========================================
echo.

call :ensure_venv
if errorlevel 1 goto :fail

call :install_packages
if errorlevel 1 goto :fail

echo Open this URL:
echo http://127.0.0.1:8765
echo.
echo Press Ctrl+C to stop.
echo.

"%PYEXE%" -u voice_dual_server.py --model "%MODEL%" --source both --device cuda --compute-type float16 --chunk-sec 15 --min-rms 0.003 --no-speech-threshold 0.8 --vad-filter %EXTRA_ARGS%
set "EXITCODE=%ERRORLEVEL%"
echo.
echo [EXIT] Code: %EXITCODE%
pause
exit /b %EXITCODE%

:ensure_venv
if exist "%PYEXE%" (
  echo [OK] Virtual environment found: %VENV%
  exit /b 0
)

echo [SETUP] Creating %VENV% ...
py -3.10 -m venv "%VENV%"
if errorlevel 1 (
  if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -m venv "%VENV%"
)
if errorlevel 1 (
  python -m venv "%VENV%"
)
if errorlevel 1 (
  echo [ERROR] Could not create virtual environment.
  exit /b 1
)
exit /b 0

:install_packages
echo [CHECK] Checking required packages ...
"%PYEXE%" -c "import numpy, sounddevice, soundcard, faster_whisper" >nul 2>nul
if not errorlevel 1 (
  echo [OK] Required packages already installed.
  exit /b 0
)

echo [SETUP] Installing required packages ...
"%PYEXE%" -m pip install numpy sounddevice soundcard faster-whisper
if errorlevel 1 exit /b 1
exit /b 0

:fail
echo.
echo [ERROR] Setup failed.
pause
exit /b 1

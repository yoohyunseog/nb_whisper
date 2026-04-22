@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV=.venv_stt"
set "PYEXE=%VENV%\Scripts\python.exe"
set "MODEL=base"
set "EXTRA_ARGS="

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
echo  Korean Mic To Text - Auto Start
echo ========================================
echo.
echo This file will:
echo  1. create/use virtual environment
echo  2. install/update required packages
echo  3. run a quick device self-test
echo  4. start Korean mic/system sound STT with Ollama DeepSeek commentary
echo.

call :ensure_venv
if errorlevel 1 goto :fail

call :install_packages
if errorlevel 1 goto :fail

call :self_test
if errorlevel 1 goto :fail

echo.
echo ========================================
echo  Korean Mic To Text - Live
echo ========================================
echo Model: %MODEL%
echo AI provider: Ollama
echo AI model: deepseek-v3.1:671b-cloud
echo Source: mic + system sound
echo Extra options:%EXTRA_ARGS%
echo First run may download the Whisper model.
echo Speak Korean into the microphone.
echo Press Ctrl+C to stop.
echo.

"%PYEXE%" -u korean_mic_stt.py --model "%MODEL%" --source both --chunk-sec 4 --no-speech-threshold 0.25 --ai-correct --ai-provider ollama --ai-mode explain --ai-model "deepseek-v3.1:671b-cloud" --ai-timeout 60 %EXTRA_ARGS%
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
  echo [WARN] py -3.10 failed. Trying py -3 ...
  py -3 -m venv "%VENV%"
)
if errorlevel 1 (
  echo [WARN] py -3 failed. Trying existing .venv Python ...
  if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -m venv "%VENV%"
)
if errorlevel 1 (
  echo [WARN] Python launcher failed. Trying python ...
  python -m venv "%VENV%"
)
if errorlevel 1 (
  echo [ERROR] Could not create virtual environment.
  echo Install Python 3.10 or newer, then run this file again.
  exit /b 1
)

echo [OK] Virtual environment created.
exit /b 0

:install_packages
echo.
echo [CHECK] Checking required packages ...
"%PYEXE%" -c "import numpy, sounddevice, soundcard, faster_whisper" >nul 2>nul
if not errorlevel 1 (
  echo [OK] Required packages already installed.
  exit /b 0
)

echo.
echo [SETUP] Upgrading pip ...
"%PYEXE%" -m pip install -U pip
if errorlevel 1 exit /b 1

echo.
echo [SETUP] Installing required packages ...
"%PYEXE%" -m pip install numpy sounddevice soundcard faster-whisper
if errorlevel 1 exit /b 1

echo [OK] Packages ready.
exit /b 0

:self_test
echo.
echo [TEST] Checking microphone/STT runtime ...
"%PYEXE%" -u korean_mic_stt.py --self-test
if errorlevel 1 exit /b 1
echo [OK] Self-test passed.
exit /b 0

:fail
echo.
echo [ERROR] Auto setup or test failed.
echo Try running this file again after checking Python, internet, and microphone permissions.
pause
exit /b 1

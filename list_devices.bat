@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Virtual environment not found. Running setup first...
  call "%~dp0setup_venv.bat"
  if errorlevel 1 exit /b 1
)

echo ========================================
echo  MyVoice Live Filter DEV - audio devices
echo ========================================
echo.

".venv\Scripts\python.exe" myvoice_live_filter.py --list-devices
set EXITCODE=%ERRORLEVEL%
echo.
pause
exit /b %EXITCODE%

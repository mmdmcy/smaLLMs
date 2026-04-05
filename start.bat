@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 start.py %*
  exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
  python start.py %*
  exit /b %errorlevel%
)

echo Python 3 was not found. Install Python 3, then run this launcher again.
exit /b 1

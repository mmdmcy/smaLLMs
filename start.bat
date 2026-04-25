@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto use_venv

where py >nul 2>nul
if not errorlevel 1 goto use_py

where python >nul 2>nul
if not errorlevel 1 goto use_python

echo Python 3 was not found. Install Python 3, then run this launcher again.
exit /b 1

:use_venv
".venv\Scripts\python.exe" start.py %*
exit /b %errorlevel%

:use_py
py -3 start.py %*
exit /b %errorlevel%

:use_python
python start.py %*
exit /b %errorlevel%

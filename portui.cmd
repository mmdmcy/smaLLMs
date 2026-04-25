@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0\.portui-runtime\portui.ps1" -ManifestDir "%~dp0portui" %*
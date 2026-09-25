@echo off
REM run.bat - one-shot launcher for the Falcon Eye / Eagle Eye recognition scanner.
REM Usage:  run.bat        (uses camera index 2 - working webcam on this machine)
REM         run.bat 0      (built-in IR camera)
REM         run.bat 1      (secondary camera)

set FALCON_CAM_INDEX=%1
if "%FALCON_CAM_INDEX%"=="" set FALCON_CAM_INDEX=2

echo Using camera index %FALCON_CAM_INDEX%

call ".venv\Scripts\activate.bat" 2>nul || echo Warning: venv not found, using system python

python -m src.recognize

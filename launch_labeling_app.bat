@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON=C:\Users\qweop\AppData\Local\Programs\Python\Python312\python.exe"
cd /d "%SCRIPT_DIR%"
start "YOLO Labeling" "%PYTHON%" "%SCRIPT_DIR%run_app.py"

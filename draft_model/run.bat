@echo off
rem Set up the environment, train the subject model and write predictions.jsonl.
rem prepare.py must have written data\ first (see README.md).
rem Double-click this file, or run it in a terminal. Run it again to continue
rem after stopping: training resumes from its last checkpoint.
setlocal
cd /d "%~dp0"
rem keep the downloaded model next to the scripts
set "HF_HOME=%~dp0hf"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "PYLAUNCH="
where py >nul 2>nul && set "PYLAUNCH=py -3"
if not defined PYLAUNCH where python >nul 2>nul && set "PYLAUNCH=python"
if not defined PYLAUNCH (
  echo Python was not found. Install Python 3.12 from python.org, then run this again.
  goto failed
)
if not exist data\valid.jsonl.gz (
  echo No training data: run prepare.py first, see README.md.
  goto failed
)
%PYLAUNCH% setup_env.py
if errorlevel 1 goto failed
if exist runs\latest.txt (
  .venv\Scripts\python train_t5.py --resume
) else (
  .venv\Scripts\python train_t5.py
)
if errorlevel 130 goto stopped
if errorlevel 1 goto failed
.venv\Scripts\python generate_t5.py
if errorlevel 1 goto failed
echo.
echo All done: predictions.jsonl is ready.
pause
exit /b 0
:stopped
echo.
echo Training stopped and saved. Run this file again to continue.
pause
exit /b 0
:failed
echo.
echo Something failed. The logs are in this folder: setup.log, runs\train.log, runs\error.log, generate_error.log
pause
exit /b 1

@echo off
rem Set up the environment, train the type model and write its predictions
rem for the held-out commits into type_predictions\. prepare_type.py must have
rem written type_data\ first, and run.bat trained the subject model, whose
rem encoder the type model starts from (see README.md).
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
if not exist type_data\valid.jsonl.gz (
  echo No training data: run prepare_type.py first, see README.md.
  goto failed
)
%PYLAUNCH% setup_env.py
if errorlevel 1 goto failed
if exist type_runs\latest.txt (
  .venv\Scripts\python train_type.py --resume
) else if exist runs\latest.txt (
  .venv\Scripts\python train_type.py --base runs
) else (
  .venv\Scripts\python train_type.py
)
if "%errorlevel%"=="130" goto stopped
if errorlevel 1 goto failed
.venv\Scripts\python predict_type.py
if errorlevel 1 goto failed
echo.
echo All done: the predictions are in type_predictions\.
pause
exit /b 0
:stopped
echo.
echo Training stopped and saved. Run this file again to continue.
pause
exit /b 0
:failed
echo.
echo Something failed. The logs are in this folder: setup.log, type_runs\train.log, type_runs\error.log, type_predict_error.log
pause
exit /b 1

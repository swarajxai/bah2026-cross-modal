@echo off
REM ============================================================
REM  Cross-Modal Satellite Image Retrieval - Pipeline runner
REM  Usage: double-click this file, or run from a terminal.
REM ============================================================

setlocal
set ROOT=%~dp0..
set VENV_PY=D:\BAH2026\.venv\Scripts\python.exe
set PROJ=D:\BAH2026\cross_modal_retrieval

echo.
echo ===========================================
echo  Cross-Modal Satellite Image Retrieval
echo ===========================================
echo.

if "%1"=="extract" goto extract
if "%1"=="train"   goto train
if "%1"=="index"   goto index
if "%1"=="eval"    goto eval
if "%1"=="web"     goto web
if "%1"=="all"     goto all
goto menu

:menu
echo Choose an action:
echo   extract  - extract backbone features
echo   train    - train modality projectors
echo   index    - build FAISS gallery index
echo   eval     - run evaluation
echo   web      - launch demo Flask server
echo   all      - run extract + train + index + eval
echo.
set /p ACTION="Action: "
goto %ACTION%

:extract
"%VENV_PY%" -m scripts.extract_features --backbone resnet50 --batch_size 32
goto end

:train
"%VENV_PY%" -m scripts.train_projectors --epochs 12
goto end

:index
"%VENV_PY%" -m scripts.build_index
goto end

:eval
"%VENV_PY%" -m scripts.evaluate
goto end

:web
cd /d "%PROJ%"
"%VENV_PY%" webapp\app.py
goto end

:all
"%VENV_PY%" -m scripts.run_all
goto end

:end
echo.
echo Done.
endlocal

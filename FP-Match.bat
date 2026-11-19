@echo off
rem FP-Match - double-click to start on Windows.
rem The first run installs the Python packages into .venv (a few minutes, needs internet).
setlocal
title FP-Match
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto deps

set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1 && set "PY=python"
if not defined PY goto nopython

echo First run: setting up FP-Match. This takes a few minutes...
if exist .venv rmdir /s /q .venv
%PY% -m venv .venv
if errorlevel 1 goto fail

:deps
fc /b requirements.txt .venv\requirements.installed >nul 2>&1
if not errorlevel 1 goto run
echo Installing / updating packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
copy /y requirements.txt .venv\requirements.installed >nul

:run
".venv\Scripts\python.exe" launcher.py
if errorlevel 1 pause
exit /b 0

:nopython
echo.
echo Python 3.9 or newer is needed.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click FP-Match.bat again.
echo.
pause
exit /b 1

:fail
echo.
echo Setup failed. Check your internet connection and try again.
pause
exit /b 1

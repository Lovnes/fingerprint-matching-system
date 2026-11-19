@echo off
rem Builds dist\FP-Match\FP-Match.exe and dist\FP-Match-Windows.zip
setlocal
cd /d "%~dp0\.."
set "PY=py -3"
py -3 --version >nul 2>&1 || set "PY=python"
if defined PYTHON set "PY=%PYTHON%"
%PY% -m venv .build-venv || exit /b 1
".build-venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
".build-venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller || exit /b 1
".build-venv\Scripts\pyinstaller.exe" --noconfirm --clean --distpath dist --workpath build packaging\fpmatch.spec || exit /b 1
if exist dist\FP-Match-Windows.zip del dist\FP-Match-Windows.zip
powershell -NoProfile -Command "Compress-Archive -Path dist\FP-Match -DestinationPath dist\FP-Match-Windows.zip" || exit /b 1
echo Built dist\FP-Match\FP-Match.exe and dist\FP-Match-Windows.zip

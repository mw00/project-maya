@echo off
rem Project Maya for Windows: the first run checks the PC, compiles the engine, asks before it
rem downloads the model, builds the pack and starts the dashboard; later runs just start it.
rem START-MAYA.bat --help lists the options. Like ./maya.sh, nothing is installed system-wide: a missing Python,
rem compiler or CUDA toolkit is reported with where to get it.
setlocal
title Project Maya
cd /d "%~dp0"
rem a .venv from an earlier run that failed half-way has a python but no pip: start it again
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pip --version >nul 2>nul || rmdir /s /q .venv
)
if exist ".venv\Scripts\python.exe" goto run

call :findpy
if defined PY goto venv
echo.
echo  64-bit Python 3.10 or newer is needed. Install it, then double-click START-MAYA.bat again:
echo    winget install -e --id Python.Python.3.12 --scope user
echo  or https://www.python.org/downloads/ (tick "Add python.exe to PATH").
echo.
pause
exit /b 1

:venv
rem a private environment inside this folder, so nothing is installed into the system Python
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto run
echo  Could not create the Python environment in .venv
pause
exit /b 1

:run
".venv\Scripts\python.exe" maya.py %*
if errorlevel 1 pause
exit /b

:findpy
rem the py launcher first, then python on PATH (not the Microsoft Store stub), then the usual per-user folders
set "PY="
py -3 -c "import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3" & goto :eof
python -c "import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) and sys.maxsize > 2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python" & goto :eof
for %%V in (313 312 311 310) do if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"" & goto :eof
goto :eof

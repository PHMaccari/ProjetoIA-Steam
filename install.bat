@echo off
cd /d "%~dp0"

set "PYTHON="
where py >nul 2>&1 && (
    py -3 -c "import sys" >nul 2>&1 && set "PYTHON=py -3"
)
if not defined PYTHON (
    where python >nul 2>&1 && (
        python -c "import sys" >nul 2>&1 && set "PYTHON=python"
    )
)
if not defined PYTHON (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    )
)

if not defined PYTHON (
    echo Python 3 nao encontrado.
    echo Instale Python 3.10+ e marque "Add to PATH".
    pause
    exit /b 1
)

%PYTHON% -m pip install -r requirements.txt
echo.
echo Instalacao concluida. Agora execute: run_app.bat
pause

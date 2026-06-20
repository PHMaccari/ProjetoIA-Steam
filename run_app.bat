@echo off
cd /d "%~dp0"

REM Tenta encontrar Python: py launcher > python no PATH > caminho padrao Windows
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
    echo Instale Python 3.10+ e marque "Add to PATH", ou ajuste este script.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo app.py nao encontrado. Execute run_app.bat dentro da pasta do projeto.
    pause
    exit /b 1
)

if not exist "steam_dataset\steamds.csv" (
    echo Dataset nao encontrado: steam_dataset\steamds.csv
    echo Baixe o arquivo e coloque nessa pasta antes de continuar.
    pause
    exit /b 1
)

if not exist "models\random_forest.joblib" (
    echo Modelo nao treinado. Treinando agora...
    %PYTHON% -m src.train
    if errorlevel 1 (
        echo Falha no treinamento.
        pause
        exit /b 1
    )
)

echo Abrindo interface Streamlit...
%PYTHON% -m streamlit run app.py

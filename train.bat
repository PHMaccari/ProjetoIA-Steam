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

if not exist "src\train.py" (
    echo src\train.py nao encontrado. Execute train.bat dentro da pasta do projeto.
    pause
    exit /b 1
)

if not exist "steam_dataset\steamds.csv" (
    echo Dataset nao encontrado: steam_dataset\steamds.csv
    echo Baixe o arquivo e coloque nessa pasta antes de continuar.
    pause
    exit /b 1
)

echo Treinando modelo Random Forest...
echo.

%PYTHON% -m src.train
if errorlevel 1 (
    echo.
    echo Falha no treinamento.
    pause
    exit /b 1
)

echo.
echo Treinamento concluido!
echo   - models\random_forest.joblib
echo   - artifacts\model_config.json
echo   - artifacts\training_report.txt
echo   - artifacts\training_dataset.parquet
echo.
echo Para abrir a interface, execute: run_app.bat
pause

@echo off
setlocal
cd /d "%~dp0"
REM Se usa Python 3.12 para coincidir con el entorno validado.
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Instala Python 3.12 con el launcher py y vuelve a ejecutar este archivo.
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m ipykernel install --user --name credit-payment-mlops --display-name "Python - Credit Payment MLOps"
if errorlevel 1 exit /b 1
echo Entorno listo. Ejecuta: .venv\Scripts\python.exe run_pipeline.py
endlocal

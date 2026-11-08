$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonDermato = Join-Path $PSScriptRoot '.venv_dermatoai\Scripts\python.exe'
if (-not (Test-Path $pythonDermato)) {
    $pythonDermato = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path $pythonDermato)) { throw 'Ejecute primero PREPARAR_DERMATOAI.ps1.' }
& $pythonDermato -m streamlit run app_dermatoai.py --server.address 127.0.0.1

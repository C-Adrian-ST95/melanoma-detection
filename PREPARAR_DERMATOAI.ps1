$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Entorno nuevo para no alterar el entorno de entrenamiento del repositorio.
if (-not (Test-Path '.venv_dermatoai\Scripts\python.exe')) {
    py -3.12 -m venv .venv_dermatoai
    if ($LASTEXITCODE -ne 0) { throw 'Instale Python 3.12 y vuelva a ejecutar.' }
}
$pythonDermato = Join-Path $PSScriptRoot '.venv_dermatoai\Scripts\python.exe'
& $pythonDermato -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'No se pudo preparar pip.' }
& $pythonDermato -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar PyTorch CPU.' }
& $pythonDermato -m pip install -r requirements-dermatoai-v2.txt
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias.' }
& $pythonDermato -m scripts_dermatoai.check_v2
if ($LASTEXITCODE -ne 0) { throw 'Revise los archivos originales ausentes o modificados.' }
Write-Host 'Preparacion terminada. Ejecute INICIAR_DERMATOAI.ps1.'

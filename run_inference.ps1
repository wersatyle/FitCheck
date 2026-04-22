param(
    [ValidateSet('lip','atr','pascal')]
    [string]$Dataset = 'lip',
    [Parameter(Mandatory=$true)]
    [string]$InputDir,
    [string]$OutputDir = './output',
    [string]$ModelRestore = '',
    [string]$Gpu = '0',
    [switch]$Logits
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot 'venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found at: $pythonExe"
}

if (-not (Test-Path $InputDir)) {
    throw "Input directory not found: $InputDir"
}

$cmd = @(
    'simple_extractor.py',
    '--dataset', $Dataset,
    '--gpu', $Gpu,
    '--input-dir', $InputDir,
    '--output-dir', $OutputDir
)

if ($ModelRestore -ne '') {
    $cmd += @('--model-restore', $ModelRestore)
}

if ($Logits.IsPresent) {
    $cmd += '--logits'
}

& $pythonExe $cmd

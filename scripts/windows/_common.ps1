Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
}

function Get-RepoPythonCommand {
    $activeEnvName = $env:CONDA_DEFAULT_ENV
    if (
        -not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX) -and
        -not [string]::IsNullOrWhiteSpace($activeEnvName) -and
        $activeEnvName -ne 'base'
    ) {
        $activePython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $activePython) {
            return ,@($activePython)
        }
    }

    $condaCommand = Get-Command "conda.exe" -ErrorAction SilentlyContinue
    if ($null -eq $condaCommand) {
        $condaCommand = Get-Command "conda" -ErrorAction SilentlyContinue
    }

    if ($null -ne $condaCommand) {
        try {
            $envListJson = & $condaCommand.Source env list --json | Out-String
            $envList = $envListJson | ConvertFrom-Json
            $gpuEnvPath = $envList.envs | Where-Object { $_ -match '[\\/]envs[\\/]gpu$' } | Select-Object -First 1
            if (-not [string]::IsNullOrWhiteSpace($gpuEnvPath)) {
                return ,@($condaCommand.Source, 'run', '-p', $gpuEnvPath, '--no-capture-output', 'python')
            }
        }
        catch {
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $activePython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $activePython) {
            return ,@($activePython)
        }
    }

    return ,@('python')
}

function Invoke-RepoPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Module,

        [string[]]$Arguments = @()
    )

    $repoRoot = Get-RepoRoot
    $srcPath = Join-Path $repoRoot "src"
    $previousPythonPath = $env:PYTHONPATH
    $pythonCommand = Get-RepoPythonCommand

    Push-Location $repoRoot
    try {
        if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
            $env:PYTHONPATH = $srcPath
        }
        else {
            $env:PYTHONPATH = "$srcPath;$previousPythonPath"
        }

        if ($pythonCommand.Length -gt 1) {
            & $pythonCommand[0] @($pythonCommand[1..($pythonCommand.Length - 1)]) -m $Module @Arguments
        }
        else {
            & $pythonCommand[0] -m $Module @Arguments
        }
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
        Pop-Location
    }
}
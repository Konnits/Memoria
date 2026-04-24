param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$OutputName
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    if ([string]::IsNullOrWhiteSpace($OutputName)) {
        throw 'Debes indicar un nombre de salida para el PDF.'
    }

    if ([System.IO.Path]::GetExtension($OutputName) -eq '') {
        $OutputName = "$OutputName.pdf"
    }

    $outputPath = if ([System.IO.Path]::IsPathRooted($OutputName)) {
        $OutputName
    } else {
        Join-Path $scriptDir $OutputName
    }

    $outputDir = Split-Path -Parent $outputPath
    if (-not [string]::IsNullOrWhiteSpace($outputDir) -and -not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }

    $commands = @(
        @{ Name = 'pdflatex (1/3)'; Command = 'pdflatex -interaction=nonstopmode main.tex' },
        @{ Name = 'biber'; Command = 'biber main' },
        @{ Name = 'pdflatex (2/3)'; Command = 'pdflatex -interaction=nonstopmode main.tex' },
        @{ Name = 'pdflatex (3/3)'; Command = 'pdflatex -interaction=nonstopmode main.tex' }
    )

    foreach ($step in $commands) {
        Write-Host "==> Ejecutando $($step.Name)..."
        Invoke-Expression $step.Command
        if ($LASTEXITCODE -ne 0) {
            throw "Falló $($step.Name) con código $LASTEXITCODE."
        }
    }

    if (-not (Test-Path 'main.pdf')) {
        throw 'No se generó main.pdf.'
    }

    $auxPatterns = @(
        'main.aux',
        'main.bbl',
        'main.bcf',
        'main.blg',
        'main.lof',
        'main.log',
        'main.lot',
        'main.out',
        'main.run.xml',
        'main.toc'
    )

    if ((Resolve-Path 'main.pdf').Path -ne [System.IO.Path]::GetFullPath($outputPath)) {
        if (Test-Path $outputPath) {
            Remove-Item $outputPath -Force
        }
        Move-Item 'main.pdf' $outputPath -Force
    }

    foreach ($pattern in $auxPatterns) {
        if (Test-Path $pattern) {
            Remove-Item $pattern -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host "PDF generado en: $outputPath"
}
finally {
    Pop-Location
}
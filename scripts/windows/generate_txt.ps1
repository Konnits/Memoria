. (Join-Path $PSScriptRoot "_common.ps1")

$repoRoot = Get-RepoRoot
$folders = @("src", "configs", "scripts")
$outputFile = Join-Path $repoRoot "ALL_FILES_CONTENT.txt"

Set-Content -Path $outputFile -Value ""
Write-Host "Generando $outputFile ..."

foreach ($folder in $folders) {
    $folderPath = Join-Path $repoRoot $folder

    Get-ChildItem -Path $folderPath -Recurse -File |
        Where-Object { $_.Extension -in @(".py", ".yaml", ".sh", ".ps1") } |
        Sort-Object FullName |
        ForEach-Object {
            $relativePath = [System.IO.Path]::GetRelativePath($repoRoot, $_.FullName).Replace('\\', '/')
            Write-Host "Procesando $relativePath ..."

            Add-Content -Path $outputFile -Value $relativePath
            Add-Content -Path $outputFile -Value '```'
            Add-Content -Path $outputFile -Value (Get-Content -Path $_.FullName -Raw)
            Add-Content -Path $outputFile -Value ""
            Add-Content -Path $outputFile -Value '```'
            Add-Content -Path $outputFile -Value ""
        }
}

Write-Host "Generacion completada: $outputFile"
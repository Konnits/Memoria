[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExperimentName,

    [Parameter(Mandatory = $true)]
    [int]$HistoryStartIndex,

    [Parameter(Mandatory = $true)]
    [int]$HistoryEndIndex,

    [string]$FutureIndexes = "",

    [string]$FutureTimestamps = "",

    [string]$FutureOffsets = "",

    [string]$Device = "cpu",

    [string]$OutputCsv = ""
)

. (Join-Path $PSScriptRoot "_common.ps1")

$arguments = @(
    "--experiment-dir", "experiments/$ExperimentName",
    "--csv-path", "data/processed/real_data_1.csv",
    "--history-start-index", $HistoryStartIndex.ToString(),
    "--history-end-index", $HistoryEndIndex.ToString(),
    "--device", $Device
)

if (-not [string]::IsNullOrWhiteSpace($FutureIndexes)) {
    $arguments += @("--future-indexes", $FutureIndexes)
}

if (-not [string]::IsNullOrWhiteSpace($FutureTimestamps)) {
    $arguments += @("--future-timestamps", $FutureTimestamps)
}

if (-not [string]::IsNullOrWhiteSpace($FutureOffsets)) {
    $arguments += @("--future-offsets", $FutureOffsets)
}

if (-not [string]::IsNullOrWhiteSpace($OutputCsv)) {
    $arguments += @("--output-csv", $OutputCsv)
}

Invoke-RepoPython -Module "ts_transformer.predict_experiment" -Arguments $arguments
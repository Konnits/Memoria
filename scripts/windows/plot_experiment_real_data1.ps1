[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExperimentName,

    [int]$MaxPoints = 1000
)

. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-RepoPython -Module "ts_transformer.plot_test_predictions" -Arguments @(
    "--data-config", "configs/data/finetune_real_data1.yaml",
    "--experiment-dir", "experiments/$ExperimentName",
    "--max-points", $MaxPoints.ToString()
)
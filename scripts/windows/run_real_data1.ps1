[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("pretrain", "finetune")]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$ExperimentName,

    [string]$InitCheckpoint = ""
)

. (Join-Path $PSScriptRoot "_common.ps1")

$modelConfig = "configs/model/transformer_base_real_data1.yaml"

if ($Stage -eq "pretrain") {
    Invoke-RepoPython -Module "ts_transformer.train" -Arguments @(
        "--stage", "pretrain",
        "--experiment-name", $ExperimentName,
        "--model-config", $modelConfig,
        "--data-config", "configs/data/pretrain_real_data1.yaml",
        "--training-config", "configs/training/pretrain_real_data1.yaml"
    )
    return
}

if ([string]::IsNullOrWhiteSpace($InitCheckpoint)) {
    $InitCheckpoint = "experiments/$ExperimentName/pretrained_model.pt"
}

Invoke-RepoPython -Module "ts_transformer.train" -Arguments @(
    "--stage", "finetune",
    "--experiment-name", $ExperimentName,
    "--init-checkpoint", $InitCheckpoint,
    "--model-config", $modelConfig,
    "--data-config", "configs/data/finetune_real_data1.yaml",
    "--training-config", "configs/training/finetune_real_data1.yaml"
)
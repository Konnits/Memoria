[CmdletBinding()]
param(
    [ValidateSet("pretrain", "finetune")]
    [string]$Stage = "finetune",

    [string]$SearchConfig = "",

    [string]$OutputDir = "experiments/hparam_search",

    [string]$InitCheckpoint = ""
)

. (Join-Path $PSScriptRoot "_common.ps1")

$trainingConfig = "configs/training/finetune_real_data1.yaml"
$dataConfig = "configs/data/finetune_real_data1.yaml"

if ($Stage -eq "pretrain") {
    $trainingConfig = "configs/training/pretrain_real_data1.yaml"
    $dataConfig = "configs/data/pretrain_real_data1.yaml"
}

if ($Stage -eq "finetune" -and [string]::IsNullOrWhiteSpace($InitCheckpoint)) {
    $defaultInitCheckpoint = "experiments/pretrain_real_data1/pretrained_model.pt"
    if (Test-Path $defaultInitCheckpoint) {
        $InitCheckpoint = $defaultInitCheckpoint
    }
}

if ([string]::IsNullOrWhiteSpace($SearchConfig)) {
    if ($Stage -eq "pretrain") {
        $SearchConfig = "configs/search/pretrain_real_data1.yaml"
    }
    else {
        $SearchConfig = "configs/search/finetune_real_data1.yaml"
    }
}

$arguments = @(
    "--stage", $Stage,
    "--search-config", $SearchConfig,
    "--model-config", "configs/model/transformer_base_real_data1.yaml",
    "--training-config", $trainingConfig,
    "--data-config", $dataConfig,
    "--output-dir", $OutputDir
)

if (-not [string]::IsNullOrWhiteSpace($InitCheckpoint)) {
    $arguments += @("--init-checkpoint", $InitCheckpoint)
}

Invoke-RepoPython -Module "ts_transformer.hyperparameter_search" -Arguments $arguments
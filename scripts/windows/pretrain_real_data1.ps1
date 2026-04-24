. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-RepoPython -Module "ts_transformer.train" -Arguments @(
    "--stage", "pretrain",
    "--model-config", "configs/model/transformer_base_real_data1.yaml",
    "--data-config", "configs/data/pretrain_real_data1.yaml",
    "--training-config", "configs/training/pretrain_real_data1.yaml"
)
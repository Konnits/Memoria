. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-RepoPython -Module "ts_transformer.train" -Arguments @(
    "--model-config", "configs/model/transformer_base_real_data1.yaml",
    "--data-config", "configs/data/real_data1.yaml",
    "--training-config", "configs/training/default_real_data1.yaml"
)
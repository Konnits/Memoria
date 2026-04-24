. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-RepoPython -Module "ts_transformer.train" -Arguments @(
    "--model-config", "configs/model/transformer_base_test1.yaml",
    "--data-config", "configs/data/toy_example_test1.yaml",
    "--training-config", "configs/training/default_test1.yaml"
)
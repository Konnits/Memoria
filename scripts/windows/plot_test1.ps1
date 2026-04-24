. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-RepoPython -Module "ts_transformer.plot_test_predictions" -Arguments @(
    "--data-config", "configs/data/toy_example_test1.yaml",
    "--experiment-dir", "experiments/exp_default"
)
. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-RepoPython -Module "ts_transformer.plot_test_predictions" -Arguments @(
    "--data-config", "configs/data/real_data1.yaml",
    "--experiment-dir", "experiments/exp_real_data1"
)
. (Join-Path $PSScriptRoot "_common.ps1")

Invoke-RepoPython -Module "ts_transformer.train" -Arguments @(
    "--stage", "finetune",
    "--init-checkpoint", "experiments/pretrain_real_data1/pretrained_model.pt",
    "--model-config", "configs/model/transformer_base_real_data1.yaml",
    "--data-config", "configs/data/finetune_real_data1.yaml",
    "--training-config", "configs/training/finetune_real_data1.yaml"
)
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

python -m ts_transformer.train \
  --stage finetune \
  --init-checkpoint experiments/pretrain_real_data1/pretrained_model.pt \
  --model-config configs/model/transformer_base_real_data1.yaml \
  --data-config configs/data/finetune_real_data1.yaml \
  --training-config configs/training/finetune_real_data1.yaml

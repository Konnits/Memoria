#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

python -m ts_transformer.train \
  --stage pretrain \
  --model-config configs/model/transformer_base_real_data1.yaml \
  --data-config configs/data/pretrain_real_data1.yaml \
  --training-config configs/training/pretrain_real_data1.yaml

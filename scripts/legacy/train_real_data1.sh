!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

python -m ts_transformer.train \
  --model-config configs/model/transformer_base_real_data1.yaml \
  --data-config configs/data/real_data1.yaml \
  --training-config configs/training/default_real_data1.yaml
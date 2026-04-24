!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

python -m ts_transformer.train \
  --model-config configs/model/transformer_base_test1.yaml \
  --data-config configs/data/toy_example_test1.yaml \
  --training-config configs/training/default_test1.yaml
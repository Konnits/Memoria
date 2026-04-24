#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

python -m ts_transformer.plot_test_predictions \
  --data-config configs/data/toy_example_test1.yaml \
  --experiment-dir experiments/exp_default
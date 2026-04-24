#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   bash scripts/plot_experiment_real_data1.sh <experiment_name> [max_points]

EXPERIMENT_NAME="${1:-}"
MAX_POINTS="${2:-1000}"

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "Uso: bash scripts/plot_experiment_real_data1.sh <experiment_name> [max_points]"
  exit 1
fi

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

python -m ts_transformer.plot_test_predictions \
  --data-config configs/data/finetune_real_data1.yaml \
  --experiment-dir "experiments/${EXPERIMENT_NAME}" \
  --max-points "$MAX_POINTS"

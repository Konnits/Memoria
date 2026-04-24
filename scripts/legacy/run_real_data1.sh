#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   bash scripts/run_real_data1.sh pretrain <experiment_name>
#   bash scripts/run_real_data1.sh finetune <experiment_name> [init_checkpoint]

STAGE="${1:-}"
EXPERIMENT_NAME="${2:-}"
INIT_CKPT="${3:-}"

if [[ -z "$STAGE" || -z "$EXPERIMENT_NAME" ]]; then
  echo "Uso: bash scripts/run_real_data1.sh <pretrain|finetune> <experiment_name> [init_checkpoint]"
  exit 1
fi

if [[ "$STAGE" != "pretrain" && "$STAGE" != "finetune" ]]; then
  echo "Error: STAGE debe ser 'pretrain' o 'finetune'."
  exit 1
fi

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src"

MODEL_CFG="configs/model/transformer_base_real_data1.yaml"

if [[ "$STAGE" == "pretrain" ]]; then
  DATA_CFG="configs/data/pretrain_real_data1.yaml"
  TRAIN_CFG="configs/training/pretrain_real_data1.yaml"

  python -m ts_transformer.train \
    --stage pretrain \
    --experiment-name "$EXPERIMENT_NAME" \
    --model-config "$MODEL_CFG" \
    --data-config "$DATA_CFG" \
    --training-config "$TRAIN_CFG"

else
  DATA_CFG="configs/data/finetune_real_data1.yaml"
  TRAIN_CFG="configs/training/finetune_real_data1.yaml"

  if [[ -z "$INIT_CKPT" ]]; then
    INIT_CKPT="experiments/${EXPERIMENT_NAME}/pretrained_model.pt"
  fi

  python -m ts_transformer.train \
    --stage finetune \
    --experiment-name "$EXPERIMENT_NAME" \
    --init-checkpoint "$INIT_CKPT" \
    --model-config "$MODEL_CFG" \
    --data-config "$DATA_CFG" \
    --training-config "$TRAIN_CFG"
fi

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-l7_expert_labels}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/expert_label_pipeline}"
RUN_DIR="$OUTPUT_ROOT/${TAG}_${RUN_ID}"
DATASET_ROOT="${MEMPATH:-$REPO_ROOT/dataset}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-12288}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-4096}"
PAIRED_DATASET="${PAIRED_DATASET:-l7/vr_paired/clean=l7/vr_paired/raw}"
WINDOW_SEC="${WINDOW_SEC:-5}"
MAX_LOAD_KG="${MAX_LOAD_KG:-30}"
LOAD_STEP_KG="${LOAD_STEP_KG:-5}"
WANDB_ENTITY="${WANDB_ENTITY:-liu-cx-tsinghua-university}"
WANDB_PROJECT="${WANDB_PROJECT:-l7_mjlab}"

NPROC="$TRAIN_GPUS" PROFILE=L7_tracking LOAD_CONFIG=l7_expert TAG="$TAG" SUFFIX="$RUN_ID" \
  OUTPUT_ROOT="$OUTPUT_ROOT" STAGES=train WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_UPLOAD_CHECKPOINTS=false \
  bash train.sh "task.num_envs=$TRAIN_NUM_ENVS" \
  "wandb.entity=$WANDB_ENTITY" "wandb.project=$WANDB_PROJECT" "$@"

CFG_PATH="$RUN_DIR/train/.hydra/config.yaml"
CHECKPOINT_PATH="$RUN_DIR/train/checkpoint_final.pt"
[[ -f "$CFG_PATH" && -f "$CHECKPOINT_PATH" ]] || {
  echo "ERROR: expert cfg/checkpoint missing under $RUN_DIR/train" >&2
  exit 3
}

CFG_PATH="$CFG_PATH" CHECKPOINT_PATH="$CHECKPOINT_PATH" \
  OUTPUT_DIR="$RUN_DIR/rollout" NUM_ENVS="$ROLLOUT_NUM_ENVS" \
  WINDOW_SEC="$WINDOW_SEC" MAX_LOAD_KG="$MAX_LOAD_KG" LOAD_STEP_KG="$LOAD_STEP_KG" \
  BUILD_LABELS=false bash scripts/run_l7_window_caps_8gpu.sh

.venv/bin/python scripts/finalize_l7_window_caps.py \
  --input-glob "$RUN_DIR/rollout/shard_*.csv" \
  --dataset-root "$DATASET_ROOT" \
  --filtered-dataset-root "$RUN_DIR/dataset_filtered" \
  --output-dir "$RUN_DIR/final" \
  --output "$RUN_DIR/final/window_caps_${WINDOW_SEC}s.npz" \
  --window-sec "$WINDOW_SEC" \
  --max-load-kg "$MAX_LOAD_KG" \
  --load-step-kg "$LOAD_STEP_KG" \
  --paired-dataset "$PAIRED_DATASET"

echo "Expert checkpoint: $CHECKPOINT_PATH"
echo "Filtered MEMPATH: $RUN_DIR/dataset_filtered"
echo "WPC labels: $RUN_DIR/final/window_caps_${WINDOW_SEC}s.npz"

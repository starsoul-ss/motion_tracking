#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOAD_CONFIG="${LOAD_CONFIG:-l7_wpc}"
TAG="${TAG:-l7_track}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/pipelines}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_DIR="$OUTPUT_ROOT/${TAG}_${RUN_ID}"

if [[ "$LOAD_CONFIG" == l7_wpc && -z "${L7_WINDOW_LOAD_CAP_LABEL_PATH:-}" ]]; then
  echo "ERROR: set L7_WINDOW_LOAD_CAP_LABEL_PATH" >&2
  exit 2
fi

run_stage() {
  local stage="$1"
  local checkpoint="${2:-}"
  local stage_dir="$RUN_DIR/$stage"
  local cmd=(uv run torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py
    task=tracking task/profile=L7_tracking "+exp=$stage" "+load=$LOAD_CONFIG"
  )
  cmd+=("${@:3}")
  cmd+=("hydra.run.dir=$stage_dir" "wandb.mode=$WANDB_MODE")
  [[ -z "$checkpoint" ]] || cmd+=("checkpoint_path=$checkpoint")
  "${cmd[@]}"
  [[ -f "$stage_dir/checkpoint_final.pt" ]] || {
    echo "ERROR: missing $stage_dir/checkpoint_final.pt" >&2
    exit 3
  }
}

if [[ "$LOAD_CONFIG" == l7_expert ]]; then
  run_stage train "" "$@"
  exit
fi

run_stage train "" "$@"
run_stage adapt "$RUN_DIR/train/checkpoint_final.pt" "$@"
run_stage finetune "$RUN_DIR/adapt/checkpoint_final.pt" "$@"

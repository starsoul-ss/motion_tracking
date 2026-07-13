#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SCRIPT="${SCRIPT:-scripts/train.py}"
PROFILE="${PROFILE:-L7_tracking}"
LOAD_CONFIG="${LOAD_CONFIG:-l7_wpc}"
TAG="${TAG:-l7_track}"
SUFFIX="${SUFFIX:-$(date +%m%d.%H%M%S)}"
NPROC="${NPROC:-8}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_UPLOAD_CHECKPOINTS="${WANDB_UPLOAD_CHECKPOINTS:-false}"
if [[ -z "${STAGES+x}" ]]; then
  [[ "$LOAD_CONFIG" == "l7_expert" ]] && STAGES="train" || STAGES="train,adapt,finetune"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/pipelines}"
PIPELINE_DIR="$OUTPUT_ROOT/${TAG}_${SUFFIX}"
EXTRA_ARGS=("$@")
# ===== Global Configuration =====

if command -v uv >/dev/null 2>&1; then
  TORCHRUN=(uv run torchrun)
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  TORCHRUN=("$REPO_ROOT/.venv/bin/python" -m torch.distributed.run)
else
  TORCHRUN=(python -m torch.distributed.run)
fi

if [[ "$LOAD_CONFIG" == "l7_wpc" && -z "${L7_WINDOW_LOAD_CAP_LABEL_PATH:-}" ]]; then
  echo "ERROR: L7_WINDOW_LOAD_CAP_LABEL_PATH must point to the public L7 window-cap label file." >&2
  exit 2
fi

find_local_checkpoint() {
  local stage_dir="$1"
  if [[ ! -d "$stage_dir" ]]; then
    return 0
  fi
  find "$stage_dir" -type f -name checkpoint_final.pt -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

require_local_checkpoint() {
  local stage="$1"
  local stage_dir="$PIPELINE_DIR/$stage"
  local checkpoint
  checkpoint="$(find_local_checkpoint "$stage_dir")"
  if [[ -z "$checkpoint" ]]; then
    echo "ERROR: local final checkpoint not found for stage '$stage' under $stage_dir" >&2
    exit 3
  fi
  realpath "$checkpoint"
}

launch_stage() {
  local stage="$1"
  local wandb_id="$2"
  local checkpoint_path="${3:-}"
  local stage_dir="$PIPELINE_DIR/$stage"
  local cmd

  mkdir -p "$stage_dir"
  cmd=("${TORCHRUN[@]}" --standalone --nproc_per_node="$NPROC" "$SCRIPT"
    task=tracking "task/profile=$PROFILE" "+exp=$stage" "+load=$LOAD_CONFIG"
    "hydra.run.dir=$stage_dir"
    "wandb.mode=$WANDB_MODE"
    "wandb.id=$wandb_id"
    "+wandb.group=${TAG}_${SUFFIX}"
    "wandb.job_type=$stage"
    "wandb.upload_checkpoints=$WANDB_UPLOAD_CHECKPOINTS"
  )
  if [[ -n "$checkpoint_path" ]]; then
    cmd+=("checkpoint_path=$checkpoint_path")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  echo ">>> ${cmd[*]}"
  "${cmd[@]}"
}

run_pipeline() {
  local id_train="${TAG}_train_${SUFFIX}"
  local id_adapt="${TAG}_adapt_${SUFFIX}"
  local id_finetune="${TAG}_finetune_${SUFFIX}"
  local train_checkpoint
  local adapt_checkpoint

  if [[ ",$STAGES," == *",train,"* ]]; then
    launch_stage train "$id_train"
  fi

  if [[ ",$STAGES," == *",adapt,"* ]]; then
    train_checkpoint="$(require_local_checkpoint train)"
    launch_stage adapt "$id_adapt" "$train_checkpoint"
  fi

  if [[ ",$STAGES," == *",finetune,"* ]]; then
    adapt_checkpoint="$(require_local_checkpoint adapt)"
    launch_stage finetune "$id_finetune" "$adapt_checkpoint"
  fi
}

echo "PROFILE=$PROFILE LOAD_CONFIG=$LOAD_CONFIG NPROC=$NPROC STAGES=$STAGES WANDB_MODE=$WANDB_MODE WANDB_UPLOAD_CHECKPOINTS=$WANDB_UPLOAD_CHECKPOINTS PIPELINE_DIR=$PIPELINE_DIR"
run_pipeline

# Examples:
#   export L7_WINDOW_LOAD_CAP_LABEL_PATH=/absolute/path/to/l7_window_cap_labels.npz
#   bash train.sh
#   NPROC=1 WANDB_MODE=offline SUFFIX=smoke bash train.sh total_frames=64 task.num_envs=4

#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====

# change this to your own project if you want to log to wandb
# you also need to change wandb project name in `cfg/train.yaml` to match this
PROJECT="axell-wppr/heft"

# change this to the number of GPUs you want to use for training
# recommended: 8 GPUs, 8*8192 envs
NPROC=8
# ===== Global Configuration =====

SCRIPT="scripts/train.py"

run_pipeline() {
  local PROFILE="$1" TAG="$2" SUFFIX="$3"

  local ID_TRAIN="${TAG}_train_${SUFFIX}"
  local ID_ADAPT="${TAG}_adapt_${SUFFIX}"
  local ID_FINETUNE="${TAG}_finetune_${SUFFIX}"

  # ---------- TRAIN ----------
  cmd=(uv run torchrun --nproc_per_node="$NPROC" "$SCRIPT"
    task=tracking "task/profile=${PROFILE}" +exp=train
    wandb.id="$ID_TRAIN"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- ADAPT ----------
  cmd=(uv run torchrun --nproc_per_node="$NPROC" "$SCRIPT"
    task=tracking "task/profile=${PROFILE}" +exp=adapt
    checkpoint_path="run:${PROJECT}/${ID_TRAIN}"
    wandb.id="$ID_ADAPT"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- FINETUNE ----------
  cmd=(uv run torchrun --nproc_per_node="$NPROC" "$SCRIPT"
    task=tracking "task/profile=${PROFILE}" +exp=finetune
    checkpoint_path="run:${PROJECT}/${ID_ADAPT}"
    wandb.id="$ID_FINETUNE"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"
}

run_pipeline "G1_tracking" "track" "0112.3"

# run_pipeline "G1/G1_gentle" "gentle" "1215"
# run_pipeline "G1/G1_no_force" "noforce" "1215"
# run_pipeline "G1/G1_extreme_force" "extremeforce" "1215"

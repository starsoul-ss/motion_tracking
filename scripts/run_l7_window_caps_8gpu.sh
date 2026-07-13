#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CFG_PATH:?Set CFG_PATH to the expert cfg.yaml}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the expert checkpoint}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/window_caps}"
NUM_ENVS="${NUM_ENVS:-4096}"
WINDOW_SEC="${WINDOW_SEC:-5}"
RAMP_SEC="${RAMP_SEC:-1}"
MAX_LOAD_KG="${MAX_LOAD_KG:-30}"
LOAD_STEP_KG="${LOAD_STEP_KG:-5}"
ROOT_ERROR_THRESHOLD_M="${ROOT_ERROR_THRESHOLD_M:-0.6}"
KEYPOINT_ERROR_THRESHOLD_M="${KEYPOINT_ERROR_THRESHOLD_M:-0.3}"
JOINT_ERROR_THRESHOLD_RAD="${JOINT_ERROR_THRESHOLD_RAD:-0.5}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-5}"
LOAD_TOLERANCE_KG="${LOAD_TOLERANCE_KG:-0.05}"
BUILD_LABELS="${BUILD_LABELS:-false}"
mkdir -p "$OUTPUT_DIR"
pids=()

for shard in {0..7}; do
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" CUDA_VISIBLE_DEVICES="$shard" \
    .venv/bin/python scripts/eval_l7_window_load_caps.py \
    "$@" \
    --cfg-path "$CFG_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --output-prefix "$OUTPUT_DIR/shard_$shard" \
    --num-envs "$NUM_ENVS" \
    --window-sec "$WINDOW_SEC" \
    --ramp-sec "$RAMP_SEC" \
    --max-load-kg "$MAX_LOAD_KG" \
    --load-step-kg "$LOAD_STEP_KG" \
    --root-error-threshold-m "$ROOT_ERROR_THRESHOLD_M" \
    --keypoint-error-threshold-m "$KEYPOINT_ERROR_THRESHOLD_M" \
    --joint-error-threshold-rad "$JOINT_ERROR_THRESHOLD_RAD" \
    --tracking-error-grace-steps "$TRACKING_ERROR_GRACE_STEPS" \
    --load-tolerance-kg "$LOAD_TOLERANCE_KG" \
    --confirm-boundary-successes \
    --task-shard-index "$shard" \
    --task-shard-count 8 \
    >"$OUTPUT_DIR/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if ((status != 0)); then
  exit "$status"
fi

if [[ "$BUILD_LABELS" == true ]]; then
  .venv/bin/python scripts/build_l7_window_cap_labels.py \
    --input-glob "$OUTPUT_DIR/shard_*.csv" \
    --output "$OUTPUT_DIR/window_caps_${WINDOW_SEC}s.npz" \
    --window-sec "$WINDOW_SEC" \
    --max-load-kg "$MAX_LOAD_KG" \
    --load-step-kg "$LOAD_STEP_KG" \
    --expected-shards 8
fi

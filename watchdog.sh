#!/bin/bash
# AME-2 Auto-restart watchdog — survives carb mutex / CUDA crashes in Isaac Sim
#
# Usage:
#   ./watchdog.sh <GPU> <SEED> <VERSION> [extra_args...]
#   ./watchdog.sh 7 570 v57a
#   ./watchdog.sh 5 571 v57b
#
# The watchdog:
#   1. Finds the latest model_*.pt checkpoint in LOG_DIR
#   2. Passes --resume <ckpt> so no training progress is lost on crash
#   3. Loops until STOP_TRAINING file appears (exit 0 is NOT a stop — Isaac Sim
#      exits 0 even on CUDA/PhysX crashes, so we always restart)

GPU=${1:-7}
SEED=${2:-570}
VERSION=${3:-v57a}
shift 3 2>/dev/null
EXTRA_ARGS="$@"   # any additional --w_* overrides from caller

BASE=/home/bsrl/hongsenpang/RLbased/ame2_standalone
PYTHON=/home/bsrl/miniconda3/envs/thunder2/bin/python
LOG_DIR=$BASE/logs_${VERSION%?}/${VERSION}   # e.g. logs_v57/v57a
LOG=$LOG_DIR.log
CRASH_LOG=$BASE/watchdog_crashes.log

mkdir -p "$LOG_DIR"
echo "[$(date)] Watchdog START  GPU=$GPU  SEED=$SEED  VERSION=$VERSION" >> "$CRASH_LOG"
echo "[$(date)] Watchdog START  GPU=$GPU  SEED=$SEED  VERSION=$VERSION"

run=0
while true; do
  run=$((run + 1))

  # Find latest checkpoint (sort -V = version sort: model_200.pt > model_50.pt)
  CKPT=$(ls "$LOG_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)

  echo "[$(date)] Run $run  GPU=$GPU  CKPT=${CKPT:-none}" >> "$CRASH_LOG"

  ARGS="--num_envs 2048 --max_iterations 80000 --seed $SEED"
  ARGS="$ARGS --log_dir $LOG_DIR --headless"
  ARGS="$ARGS $EXTRA_ARGS"

  if [ -n "$CKPT" ]; then
    ARGS="$ARGS --resume $CKPT"
    echo "[$(date)] Resuming from $CKPT" >> "$CRASH_LOG"
  fi

  cd "$BASE" && \
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PYTHON scripts/train_ame2_direct.py $ARGS >> "$LOG" 2>&1

  EXIT=$?
  echo "[$(date)] Run $run exited with code $EXIT" >> "$CRASH_LOG"

  # Only stop on manual request — Isaac Sim exits 0 even on CUDA crashes
  if [ -f "$BASE/STOP_TRAINING" ]; then
    echo "[$(date)] Stop requested. Exiting watchdog." >> "$CRASH_LOG"
    break
  fi

  echo "[$(date)] Exited (code=$EXIT), restarting in 30s..." >> "$CRASH_LOG"
  sleep 30
done

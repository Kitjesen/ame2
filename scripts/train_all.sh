#!/usr/bin/env bash
# =============================================================================
# AME-2 End-to-End Training Pipeline
# =============================================================================
# Runs all three phases sequentially on a single node.
# See docs/training_walkthrough.md for full parameter explanations.
#
# Usage:
#   bash scripts/train_all.sh                        # defaults (4800 envs)
#   bash scripts/train_all.sh --num_envs 512         # single-GPU debug
#   bash scripts/train_all.sh --phase_start 1        # skip Phase 0
#   bash scripts/train_all.sh --phase_start 2 \
#       --teacher_ckpt logs/ame2_teacher_X/model_80000.pt
#
# Requirements (Phase 1 / 2):
#   - NVIDIA Isaac Sim 4.x
#   - Isaac Lab  (pip install -e source/isaaclab)
#   - isaaclab_rl, isaaclab_assets
#   - robot_lab  (pip install -e source/robot_lab)
#
# Phase 0 (MappingNet pretraining) has NO Isaac Sim dependency.
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
NUM_ENVS=4800          # 4800 × 8 GPU (paper scale); use 512-1024 for single GPU
LOG_DIR="logs"
PHASE_START=0          # Start from phase 0, 1, or 2
TEACHER_CKPT=""        # Auto-detected if empty and phase_start=2
MAPPING_STEPS=50000    # MappingNet gradient steps (100 = quick demo; 50000 = paper)
MAPPING_BATCH=64       # Batch size for MappingNet pretraining
SEED=42
DEVICE="cuda"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_envs)      NUM_ENVS="$2";      shift 2 ;;
        --log_dir)       LOG_DIR="$2";       shift 2 ;;
        --phase_start)   PHASE_START="$2";   shift 2 ;;
        --teacher_ckpt)  TEACHER_CKPT="$2";  shift 2 ;;
        --mapping_steps) MAPPING_STEPS="$2"; shift 2 ;;
        --mapping_batch) MAPPING_BATCH="$2"; shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --device)        DEVICE="$2";        shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

MAPPING_CKPT="${LOG_DIR}/mapping_net.pt"

echo "============================================================"
echo "  AME-2 Training Pipeline"
echo "  Phase start : ${PHASE_START}"
echo "  Num envs    : ${NUM_ENVS}"
echo "  Log dir     : ${LOG_DIR}"
echo "  Device      : ${DEVICE}"
echo "  Seed        : ${SEED}"
echo "============================================================"

mkdir -p "${LOG_DIR}"

# ── Phase 0: MappingNet Pretraining ──────────────────────────────────────────
if [[ ${PHASE_START} -le 0 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Phase 0: MappingNet Pretraining"
    echo "  Steps : ${MAPPING_STEPS} | Batch : ${MAPPING_BATCH}"
    echo "  Save  : ${MAPPING_CKPT}"
    echo "──────────────────────────────────────────────────────────"

    python scripts/train_mapping.py \
        --num_steps    "${MAPPING_STEPS}" \
        --batch_size   "${MAPPING_BATCH}" \
        --save_path    "${MAPPING_CKPT}" \
        --output_dir   "${LOG_DIR}/mapping_vis" \
        --vis_interval 500

    echo "[Phase 0] Done. Checkpoint: ${MAPPING_CKPT}"
fi

# ── Phase 1: Teacher PPO ──────────────────────────────────────────────────────
if [[ ${PHASE_START} -le 1 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Phase 1: Teacher PPO  (80 000 iterations)"
    echo "  Envs  : ${NUM_ENVS}"
    echo "  Map   : ${MAPPING_CKPT}"
    echo "──────────────────────────────────────────────────────────"

    python scripts/train_ame2.py \
        --phase          1 \
        --num_envs       "${NUM_ENVS}" \
        --log_dir        "${LOG_DIR}" \
        --device         "${DEVICE}" \
        --seed           "${SEED}" \
        --headless

    # Locate latest teacher checkpoint for Phase 2
    TEACHER_DIR=$(ls -td "${LOG_DIR}"/ame2_teacher_* 2>/dev/null | head -1)
    if [[ -z "${TEACHER_DIR}" ]]; then
        echo "[ERROR] No teacher checkpoint directory found in ${LOG_DIR}"
        exit 1
    fi
    TEACHER_CKPT="${TEACHER_DIR}/model_80000.pt"
    echo "[Phase 1] Done. Teacher checkpoint: ${TEACHER_CKPT}"
fi

# ── Phase 2: Student Distillation + PPO ──────────────────────────────────────
if [[ ${PHASE_START} -le 2 ]]; then
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Phase 2: Student Distillation + PPO  (40 000 iterations)"
    echo "  Teacher : ${TEACHER_CKPT}"
    echo "──────────────────────────────────────────────────────────"

    if [[ -z "${TEACHER_CKPT}" ]]; then
        echo "[ERROR] --teacher_ckpt required when --phase_start 2"
        exit 1
    fi

    python scripts/train_ame2.py \
        --phase          2 \
        --teacher_ckpt   "${TEACHER_CKPT}" \
        --num_envs       "${NUM_ENVS}" \
        --log_dir        "${LOG_DIR}" \
        --device         "${DEVICE}" \
        --seed           "${SEED}" \
        --headless

    echo "[Phase 2] Done. Student checkpoint saved under ${LOG_DIR}/ame2_student_*/"
fi

echo ""
echo "============================================================"
echo "  All phases complete."
echo "  Checkpoints under: ${LOG_DIR}/"
echo "============================================================"

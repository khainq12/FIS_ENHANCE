#!/bin/bash
#=============================================================================
#  run_all.sh (AUTO LOG VERSION)
# nohup bash run_all2.sh &
#=============================================================================

set -uo pipefail

# ================= AUTO LOG =================
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# Ghi log + vẫn hiển thị ra terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "📝 Log file: $LOG_FILE"

# tránh buffer Python
export PYTHONUNBUFFERED=1

# --------------- CẤU HÌNH ---------------
DATASET="cifar10"
IMAGE_SIZE=32
SNR_MIN=1
SNR_MAX=13
TRAIN_SNR="1 4 7 10 13"
EVAL_SNR="1 4 7 10 13"
BUDGET=1.0
RATIO=0.1667
BASE_DIR="exp_ctx"
SNR_DB=1
BASELINE_CKPT_NAME="baseline_best.pth"
FIS_CKPT_NAME="fis_power_best.pth"

# --------------- Warm-start ---------------
WARMSTART=true
WARMSTART_CONTROLLER_EPOCHS=10
FINETUNE_LR=1e-5

# --------------- Chế độ ---------------
TRAIN_ONLY=false
DIAG_ONLY=false
SIMS_ONLY=false
DRY_RUN=false

# --------------- Chọn kênh ---------------
RUN_AWGN=true
RUN_NOEQ=true
RUN_EQ=true

# --------------- Parse args ---------------
for arg in "$@"; do
    case $arg in
        --awgn)         RUN_NOEQ=false; RUN_EQ=false ;;
        --noeq)         RUN_AWGN=false; RUN_EQ=false ;;
        --eq)           RUN_AWGN=false; RUN_NOEQ=false ;;
        --rayleigh)     RUN_AWGN=false ;;
        --snr)          shift; SNR_DB="${1:-13}" ;;
        --train-only)   DIAG_ONLY=false; SIMS_ONLY=false; TRAIN_ONLY=true ;;
        --diag-only)    TRAIN_ONLY=false; SIMS_ONLY=false; DIAG_ONLY=true ;;
        --sims-only)    TRAIN_ONLY=false; DIAG_ONLY=false; SIMS_ONLY=true ;;
        --fast)         TRAIN_SNR="1 7 13"; EVAL_SNR="1 7 13" ;;
        --no-warmstart) WARMSTART=false ;;
        --dry)          DRY_RUN=true ;;
    esac
done

echo "🔧 MODE:"
[ "$TRAIN_ONLY" = true ] && echo "  → TRAIN ONLY"
[ "$DIAG_ONLY" = true ] && echo "  → DIAG ONLY"
[ "$SIMS_ONLY" = true ] && echo "  → SIMS ONLY"

[ "$WARMSTART" = true ] && echo "🔥 Warmstart ON" || echo "🔥 Warmstart OFF"

# ================================================================
# FUNCTIONS
# ================================================================

run_cmd() {
    echo ""
    echo "------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------"
    [ "$DRY_RUN" = false ] && eval "$1"
}

run_baseline() {
    local TAG=$1
    local CHANNEL=$2
    local DIR_SUFFIX="${3:-}"
    local EXTRA_FLAG="${4:-}"
    local SAVE_DIR="${BASE_DIR}/ckpts_${TAG}_baseline_${CHANNEL}${DIR_SUFFIX}"

    run_cmd "python -u train_baseline.py \
  --dataset ${DATASET} --image_size ${IMAGE_SIZE} \
  --channel ${CHANNEL} \
  --snr_min ${SNR_MIN} --snr_max ${SNR_MAX} \
  --eval_snr_list ${EVAL_SNR} \
  --save_dir ${SAVE_DIR} ${EXTRA_FLAG}"
}

run_fis() {
    local MODE=$1
    local TAG=$2
    local CHANNEL=$3
    local DIR_SUFFIX="${4:-}"
    local EXTRA_FLAG="${5:-}"
    local SAVE_DIR="${BASE_DIR}/ckpts_${TAG}_${MODE}_${CHANNEL}${DIR_SUFFIX}"

    local WARMSTART_FLAG=""
    if [ "$WARMSTART" = true ]; then
        local BASELINE_CKPT="${BASE_DIR}/ckpts_${TAG}_baseline_${CHANNEL}${DIR_SUFFIX}/${BASELINE_CKPT_NAME}"
        if [ -f "${BASELINE_CKPT}" ]; then
            WARMSTART_FLAG="--baseline_ckpt ${BASELINE_CKPT} \
  --warmstart_controller_only_epochs ${WARMSTART_CONTROLLER_EPOCHS} \
  --finetune_lr ${FINETUNE_LR}"
            echo "🔥 Warmstart from ${BASELINE_CKPT}"
        fi
    fi

    run_cmd "python -u train_fis_power.py \
  --dataset ${DATASET} --image_size ${IMAGE_SIZE} \
  --channel ${CHANNEL} --mode ${MODE} --budget ${BUDGET} \
  --snr_min ${SNR_MIN} --snr_max ${SNR_MAX} \
  --train_snr_list ${TRAIN_SNR} --eval_snr_list ${EVAL_SNR} \
  --save_dir ${SAVE_DIR} ${WARMSTART_FLAG} ${EXTRA_FLAG}"
}

run_diag() {
    local TAG=$1
    local CHANNEL=$2
    local DIR_SUFFIX="${3:-}"
    local EXTRA_FLAG="${4:-}"
    local LABEL="${5:-Rayleigh}"

    run_cmd "python -u diagnose_controller.py \
  --baseline_ckpt ${BASE_DIR}/ckpts_${TAG}_baseline_${CHANNEL}${DIR_SUFFIX}/${BASELINE_CKPT_NAME} \
  --linear_ckpt ${BASE_DIR}/ckpts_${TAG}_linear_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME} \
  --importance_only_ckpt ${BASE_DIR}/ckpts_${TAG}_importance_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME} \
  --snr_only_ckpt ${BASE_DIR}/ckpts_${TAG}_snr_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME} \
  --full_ckpt ${BASE_DIR}/ckpts_${TAG}_full_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME} \
  --channel ${LABEL} --snr_db ${SNR_DB} --ratio ${RATIO} ${EXTRA_FLAG}"
}

run_paper_sims() {
    local TAG=$1
    local CHANNEL=$2
    local DIR_SUFFIX="${3:-}"
    local EXTRA_FLAG="${4:-}"
    local LABEL="${5:-Rayleigh}"

    local MAP_FILE=".tmp_map.json"

    cat > "${MAP_FILE}" << EOF
{
  "linear": "${BASE_DIR}/ckpts_${TAG}_linear_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}",
  "importance_only": "${BASE_DIR}/ckpts_${TAG}_importance_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}",
  "snr_only": "${BASE_DIR}/ckpts_${TAG}_snr_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}",
  "full": "${BASE_DIR}/ckpts_${TAG}_full_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}"
}
EOF

    run_cmd "python -u run_paper_sims.py \
  --baseline_ckpt ${BASE_DIR}/ckpts_${TAG}_baseline_${CHANNEL}${DIR_SUFFIX}/${BASELINE_CKPT_NAME} \
  --fis_ckpt_map_json ${MAP_FILE} \
  --channel ${LABEL} \
  --ratio ${RATIO} --budget ${BUDGET} \
  --snrs 1 4 7 10 13 \
  --modes baseline,linear,importance_only,snr_only,full \
  --dataset ${DATASET} --image_size ${IMAGE_SIZE}"

    rm -f "${MAP_FILE}"
}

# ================================================================
# RUN
# ================================================================

echo "🚀 START PIPELINE"

[ "$RUN_AWGN" = true ] && {
    run_baseline "awgn" "AWGN"
    for m in linear importance_only snr_only full; do
        run_fis "$m" "awgn" "AWGN"
    done
}

[ "$RUN_NOEQ" = true ] && {
    run_baseline "noeq" "Rayleigh"
    for m in linear importance_only snr_only full; do
        run_fis "$m" "noeq" "Rayleigh"
    done
}

[ "$RUN_EQ" = true ] && {
    run_baseline "eq" "Rayleigh" "_eq" "--rayleigh_equalize"
    for m in linear importance_only snr_only full; do
        run_fis "$m" "eq" "Rayleigh" "_eq" "--rayleigh_equalize"
    done
}

echo "✅ DONE"
#!/bin/bash
#=============================================================================
#  run_all_final.sh (FULL PIPELINE + LOG + CONTROL)
#=============================================================================

set -euo pipefail

# ================= AUTO LOG =================
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "📝 Log file: $LOG_FILE"

export PYTHONUNBUFFERED=1

# ================= CONFIG =================
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

BASELINE_CKPT="baseline_best.pth"
FIS_CKPT="fis_power_best.pth"

# ================= FLAGS =================
TRAIN_ONLY=false
DIAG_ONLY=false
SIMS_ONLY=false
DRY_RUN=false

RUN_AWGN=true
RUN_NOEQ=true
RUN_EQ=true

# ================= PARSE =================
for arg in "$@"; do
    case $arg in
        --awgn) RUN_NOEQ=false; RUN_EQ=false ;;
        --noeq) RUN_AWGN=false; RUN_EQ=false ;;
        --eq) RUN_AWGN=false; RUN_NOEQ=false ;;
        --rayleigh) RUN_AWGN=false ;;
        --train-only) TRAIN_ONLY=true ;;
        --diag-only) DIAG_ONLY=true ;;
        --sims-only) SIMS_ONLY=true ;;
        --fast) TRAIN_SNR="1 7 13"; EVAL_SNR="1 7 13" ;;
        --dry) DRY_RUN=true ;;
    esac
done

echo "🚀 START PIPELINE"

run_cmd() {
    echo ""
    echo "------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------"
    [ "$DRY_RUN" = false ] && eval "$1"
}

# ================= TRAIN =================
run_baseline() {
    TAG=$1; CHANNEL=$2; SUFFIX=${3:-}; EXTRA=${4:-}

    run_cmd "python -u train_baseline.py \
        --dataset $DATASET --image_size $IMAGE_SIZE \
        --channel $CHANNEL \
        --snr_min $SNR_MIN --snr_max $SNR_MAX \
        --eval_snr_list $EVAL_SNR \
        --save_dir $BASE_DIR/ckpts_${TAG}_baseline_${CHANNEL}${SUFFIX} $EXTRA"
}

run_fis() {
    MODE=$1; TAG=$2; CHANNEL=$3; SUFFIX=${4:-}; EXTRA=${5:-}

    run_cmd "python -u train_fis_power.py \
        --dataset $DATASET --image_size $IMAGE_SIZE \
        --channel $CHANNEL --mode $MODE --budget $BUDGET \
        --snr_min $SNR_MIN --snr_max $SNR_MAX \
        --train_snr_list $TRAIN_SNR --eval_snr_list $EVAL_SNR \
        --save_dir $BASE_DIR/ckpts_${TAG}_${MODE}_${CHANNEL}${SUFFIX} $EXTRA"
}

# ================= DIAG =================
run_diag() {
    TAG=$1; CHANNEL=$2; SUFFIX=${3:-}; EXTRA=${4:-}; LABEL=${5:-$CHANNEL}

    run_cmd "python -u diagnose_controller.py \
        --baseline_ckpt $BASE_DIR/ckpts_${TAG}_baseline_${CHANNEL}${SUFFIX}/$BASELINE_CKPT \
        --linear_ckpt $BASE_DIR/ckpts_${TAG}_linear_${CHANNEL}${SUFFIX}/$FIS_CKPT \
        --importance_only_ckpt $BASE_DIR/ckpts_${TAG}_importance_only_${CHANNEL}${SUFFIX}/$FIS_CKPT \
        --snr_only_ckpt $BASE_DIR/ckpts_${TAG}_snr_only_${CHANNEL}${SUFFIX}/$FIS_CKPT \
        --full_ckpt $BASE_DIR/ckpts_${TAG}_full_${CHANNEL}${SUFFIX}/$FIS_CKPT \
        --channel $LABEL --snr_db $SNR_DB --ratio $RATIO $EXTRA"
}

# ================= PAPER SIMS =================
run_sims() {
    TAG=$1; CHANNEL=$2; SUFFIX=${3:-}; EXTRA=${4:-}; LABEL=${5:-$CHANNEL}

    MAP_FILE=".tmp_${TAG}.json"

    cat > "$MAP_FILE" << EOF
{
  "linear": "$BASE_DIR/ckpts_${TAG}_linear_${CHANNEL}${SUFFIX}/$FIS_CKPT",
  "importance_only": "$BASE_DIR/ckpts_${TAG}_importance_only_${CHANNEL}${SUFFIX}/$FIS_CKPT",
  "snr_only": "$BASE_DIR/ckpts_${TAG}_snr_only_${CHANNEL}${SUFFIX}/$FIS_CKPT",
  "full": "$BASE_DIR/ckpts_${TAG}_full_${CHANNEL}${SUFFIX}/$FIS_CKPT"
}
EOF

    run_cmd "python -u run_paper_sims.py \
        --baseline_ckpt $BASE_DIR/ckpts_${TAG}_baseline_${CHANNEL}${SUFFIX}/$BASELINE_CKPT \
        --fis_ckpt_map_json $MAP_FILE \
        --channel $LABEL \
        --ratio $RATIO --budget $BUDGET \
        --snrs 1 4 7 10 13 \
        --modes baseline,linear,importance_only,snr_only,full \
        --dataset $DATASET --image_size $IMAGE_SIZE $EXTRA"

    rm -f "$MAP_FILE"
}

# ================= RUN =================

# ---- TRAIN ----
if [ "$DIAG_ONLY" = false ] && [ "$SIMS_ONLY" = false ]; then
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
fi

# ---- DIAG ----
if [ "$TRAIN_ONLY" = false ] && [ "$SIMS_ONLY" = false ]; then
    [ "$RUN_AWGN" = true ] && run_diag "awgn" "AWGN"
    [ "$RUN_NOEQ" = true ] && run_diag "noeq" "Rayleigh"
    [ "$RUN_EQ" = true ] && run_diag "eq" "Rayleigh" "_eq" "--rayleigh_equalize"
fi

# ---- SIMS ----
if [ "$TRAIN_ONLY" = false ] && [ "$DIAG_ONLY" = false ]; then
    [ "$RUN_AWGN" = true ] && run_sims "awgn" "AWGN"
    [ "$RUN_NOEQ" = true ] && run_sims "noeq" "Rayleigh"
    [ "$RUN_EQ" = true ] && run_sims "eq" "Rayleigh" "_eq" "--rayleigh_equalize"
fi

echo "✅ DONE"
#!/bin/bash
#=============================================================================
#  run_all.sh
#  Train + Diagnose + Paper Sims cho 3 kênh:
#    AWGN | Rayleigh No-EQ | Rayleigh EQ
#
#  Dùng: bash run_all.sh [tùy chọn]
#
#  Tùy chọn kênh (mặc định: chạy cả 3):
#    --awgn        : chỉ chạy AWGN
#    --noeq        : chỉ chạy Rayleigh no-equalize
#    --eq          : chỉ chạy Rayleigh equalize
#    --rayleigh    : chạy cả Rayleigh (no-eq + eq)
#
#  Tùy chọn phần (mặc định: chạy cả 3 phần):
#    --train-only  : chỉ train (Part 1)
#    --diag-only   : chỉ diagnose (Part 2)
#    --sims-only   : chỉ paper sims PSNR/SSIM (Part 3)
#
#  Tùy chọn khác:
#    --fast        : chế độ nhanh (chỉ 3 SNR: 1 7 13)
#    --snr N       : SNR dB cho diagnose (mặc định: 13)
#    --dry         : chỉ in lệnh, không chạy thực sự
#=============================================================================

set -uo pipefail

# --------------- CẤU HÝNH ---------------
DATASET="cifar10"
IMAGE_SIZE=32
SNR_MIN=1
SNR_MAX=13
TRAIN_SNR="1 4 7 10 13"
EVAL_SNR="1 4 7 10 13"
BUDGET=1.0
RATIO=0.1667
BASE_DIR="exp_ctx"
SNR_DB=13
BASELINE_CKPT_NAME="baseline_best.pth"
FIS_CKPT_NAME="fis_power_best.pth"

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
        --snr)          shift; SNR_DB="${1:-13}" ;;
        --train-only)   DIAG_ONLY=false; SIMS_ONLY=false; TRAIN_ONLY=true ;;
        --diag-only)    TRAIN_ONLY=false; SIMS_ONLY=false; DIAG_ONLY=true ;;
        --sims-only)    TRAIN_ONLY=false; DIAG_ONLY=false; SIMS_ONLY=true ;;
        --fast)         TRAIN_SNR="1 7 13"; EVAL_SNR="1 7 13" ;;
        --dry)          DRY_RUN=true ;;
    esac
done

if [ "$DRY_RUN" = true ]; then echo "🔍 DRY RUN"; fi
if [ "$TRAIN_ONLY" = true ]; then echo "🔧 CHẾ ĐỘ: CHỈ TRAIN (Part 1)"
elif [ "$DIAG_ONLY" = true ]; then echo "🔧 CHẾ ĐỘ: CHỈ DIAGNOSE (Part 2)"
elif [ "$SIMS_ONLY" = true ]; then echo "🔧 CHẾ ĐỘ: CHỈ PAPER SIMS (Part 3)"
else echo "🔧 CHẾ ĐỘ: TRAIN + DIAGNOSE + PAPER SIMS"; fi

# ================================================================
#  HÀM
# ================================================================

# --- Part 1: Train Baseline ---
# $1=TAG  $2=CHANNEL  $3=DIR_SUFFIX  $4=EXTRA_FLAG
run_baseline() {
    local TAG=$1
    local CHANNEL=$2
    local DIR_SUFFIX="${3:-}"
    local EXTRA_FLAG="${4:-}"
    local SAVE_DIR="${BASE_DIR}/ckpts_${TAG}_baseline_${CHANNEL}${DIR_SUFFIX}"

    local CMD="python train_baseline.py \
  --dataset ${DATASET} --image_size ${IMAGE_SIZE} \
  --channel ${CHANNEL} \
  --snr_min ${SNR_MIN} --snr_max ${SNR_MAX} \
  --eval_snr_list ${EVAL_SNR} \
  --save_dir ${SAVE_DIR} ${EXTRA_FLAG}"

    echo ""
    echo "============================================================"
    echo "  🔰 [${DONE}/${TOTAL}] BASELINE ${TAG} | Channel: ${CHANNEL}"
    echo "  📁 ${SAVE_DIR}/"
    echo "============================================================"
    echo "$CMD"
    echo ""
    [ "$DRY_RUN" = false ] && eval "$CMD"
}

# --- Part 1: Train FIS ---
# $1=MODE  $2=TAG  $3=CHANNEL  $4=DIR_SUFFIX  $5=EXTRA_FLAG
run_fis() {
    local MODE=$1
    local TAG=$2
    local CHANNEL=$3
    local DIR_SUFFIX="${4:-}"
    local EXTRA_FLAG="${5:-}"
    local SAVE_DIR="${BASE_DIR}/ckpts_${TAG}_${MODE}_${CHANNEL}${DIR_SUFFIX}"

    local CMD="python train_fis_power.py \
  --dataset ${DATASET} --image_size ${IMAGE_SIZE} \
  --channel ${CHANNEL} --mode ${MODE} --budget ${BUDGET} \
  --snr_min ${SNR_MIN} --snr_max ${SNR_MAX} \
  --train_snr_list ${TRAIN_SNR} --eval_snr_list ${EVAL_SNR} \
  --save_dir ${SAVE_DIR} ${EXTRA_FLAG}"

    echo ""
    echo "============================================================"
    echo "  🚀 [${DONE}/${TOTAL}] ${MODE} (${TAG}) | Channel: ${CHANNEL}"
    echo "  📁 ${SAVE_DIR}/"
    echo "============================================================"
    echo "$CMD"
    echo ""
    [ "$DRY_RUN" = false ] && eval "$CMD"
}

# --- Part 2: Diagnose ---
# $1=TAG  $2=CHANNEL  $3=DIR_SUFFIX  $4=EXTRA_FLAG  $5=LABEL
run_diag() {
    local TAG=$1
    local CHANNEL=$2
    local DIR_SUFFIX="${3:-}"
    local EXTRA_FLAG="${4:-}"
    local LABEL="${5:-Rayleigh}"
    local SAVE_DIR="diag_${TAG}_snr${SNR_DB}"

    local CKPT_BASE="${BASE_DIR}/ckpts_${TAG}_baseline_${CHANNEL}${DIR_SUFFIX}/${BASELINE_CKPT_NAME}"
    local CKPT_LIN="${BASE_DIR}/ckpts_${TAG}_linear_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}"
    local CKPT_IMP="${BASE_DIR}/ckpts_${TAG}_importance_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}"
    local CKPT_SNR="${BASE_DIR}/ckpts_${TAG}_snr_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}"
    local CKPT_FULL="${BASE_DIR}/ckpts_${TAG}_full_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}"

    local CMD="python diagnose_controller.py \
  --baseline_ckpt ${CKPT_BASE} \
  --linear_ckpt ${CKPT_LIN} \
  --importance_only_ckpt ${CKPT_IMP} \
  --snr_only_ckpt ${CKPT_SNR} \
  --full_ckpt ${CKPT_FULL} \
  --channel ${LABEL} --snr_db ${SNR_DB} --ratio ${RATIO} ${EXTRA_FLAG} \
  --save_dir ${SAVE_DIR}"

    echo ""
    echo "============================================================"
    echo "  🔬 DIAGNOSE [${TAG}] ${LABEL} | SNR = ${SNR_DB} dB"
    echo "  📁 ${SAVE_DIR}/"
    echo "============================================================"
    echo "$CMD"
    echo ""
    [ "$DRY_RUN" = false ] && eval "$CMD"
}

# --- Part 3: Paper Sims (PSNR/SSIM toàn bộ test set) ---
# $1=TAG  $2=CHANNEL  $3=DIR_SUFFIX  $4=EXTRA_FLAG  $5=LABEL
run_paper_sims() {
    local TAG=$1
    local CHANNEL=$2
    local DIR_SUFFIX="${3:-}"
    local EXTRA_FLAG="${4:-}"
    local LABEL="${5:-Rayleigh}"
    local SAVE_DIR="paper_sims_${TAG}"

    # Tạo JSON map cho FIS checkpoints (đường dẫn chính xác)
    local MAP_FILE=".tmp_fis_map_${TAG}.json"
    cat > "${MAP_FILE}" << MAP_EOF
{
  "linear":            "${BASE_DIR}/ckpts_${TAG}_linear_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}",
  "importance_only":   "${BASE_DIR}/ckpts_${TAG}_importance_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}",
  "snr_only":          "${BASE_DIR}/ckpts_${TAG}_snr_only_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}",
  "full":              "${BASE_DIR}/ckpts_${TAG}_full_${CHANNEL}${DIR_SUFFIX}/${FIS_CKPT_NAME}"
}
MAP_EOF

    local CKPT_BASE="${BASE_DIR}/ckpts_${TAG}_baseline_${CHANNEL}${DIR_SUFFIX}/${BASELINE_CKPT_NAME}"

    local CMD="python run_paper_sims.py \
  --baseline_ckpt ${CKPT_BASE} \
  --fis_ckpt_map_json ${MAP_FILE} \
  --channel ${LABEL} \
  --ratio ${RATIO} \
  --budget ${BUDGET} \
  --snrs 1 4 7 10 13 \
  --modes baseline,linear,importance_only,snr_only,full \
  --dataset ${DATASET} --image_size ${IMAGE_SIZE} \
  --batch_size 128 \
  --save_dir ${SAVE_DIR} ${EXTRA_FLAG}"

    echo ""
    echo "============================================================"
    echo "  📊 PAPER SIMS [${TAG}] ${LABEL}"
    echo "  📁 ${SAVE_DIR}/"
    echo "============================================================"
    echo "$CMD"
    echo ""
    [ "$DRY_RUN" = false ] && eval "$CMD"

    # Cleanup temp file
    rm -f "${MAP_FILE}"
}

# ================================================================
#  PART 1 — TRAIN
# ================================================================

if [ "$DIAG_ONLY" = false ] && [ "$SIMS_ONLY" = false ]; then

    MODES=("linear" "importance_only" "snr_only" "full")
    TOTAL=0
    DONE=0
    [ "$RUN_AWGN" = true ] && TOTAL=$((TOTAL + 5))
    [ "$RUN_NOEQ" = true ] && TOTAL=$((TOTAL + 5))
    [ "$RUN_EQ" = true ]    && TOTAL=$((TOTAL + 5))

    echo ""
    echo "████████████████████████████████████████████████████████████"
    echo "  📦 PART 1 — TRAINING (${TOTAL} jobs)"
    echo "████████████████████████████████████████████████████████████"

    if [ "$RUN_AWGN" = true ]; then
        echo ""; echo "── 1️⃣  AWGN ──"
        DONE=$((DONE + 1)); run_baseline "awgn" "AWGN" "" ""
        for m in "${MODES[@]}"; do DONE=$((DONE + 1)); run_fis "$m" "awgn" "AWGN" "" ""; done
    fi

    if [ "$RUN_NOEQ" = true ]; then
        echo ""; echo "── 2️⃣  RAYLEIGH NO-EQ ──"
        DONE=$((DONE + 1)); run_baseline "noeq" "Rayleigh" "" ""
        for m in "${MODES[@]}"; do DONE=$((DONE + 1)); run_fis "$m" "noeq" "Rayleigh" "" ""; done
    fi

    if [ "$RUN_EQ" = true ]; then
        echo ""; echo "── 3️⃣  RAYLEIGH EQ ──"
        DONE=$((DONE + 1)); run_baseline "eq" "Rayleigh" "_eq" "--rayleigh_equalize"
        for m in "${MODES[@]}"; do DONE=$((DONE + 1)); run_fis "$m" "eq" "Rayleigh" "_eq" "--rayleigh_equalize"; done
    fi

    echo ""; echo "✅ TRAINING XONG"
fi

# ================================================================
#  PART 2 — DIAGNOSE
# ================================================================

if [ "$TRAIN_ONLY" = false ] && [ "$SIMS_ONLY" = false ]; then

    echo ""
    echo "████████████████████████████████████████████████████████████"
    echo "  🔬 PART 2 — DIAGNOSE (SNR = ${SNR_DB} dB)"
    echo "████████████████████████████████████████████████████████████"

    [ "$RUN_AWGN" = true ] && run_diag "awgn" "AWGN" "" "" "AWGN"
    [ "$RUN_NOEQ" = true ] && run_diag "noeq" "Rayleigh" "" "" "Rayleigh"
    [ "$RUN_EQ" = true ]    && run_diag "eq" "Rayleigh" "_eq" "--rayleigh_equalize" "Rayleigh"

    echo ""
    echo "✅ DIAGNOSE XONG"
    [ "$RUN_AWGN" = true ] && echo "  📂 diag_awgn_snr${SNR_DB}/"
    [ "$RUN_NOEQ" = true ] && echo "  📂 diag_noeq_snr${SNR_DB}/"
    [ "$RUN_EQ" = true ]    && echo "  📂 diag_eq_snr${SNR_DB}/"
fi

# ================================================================
#  PART 3 — PAPER SIMS (PSNR/SSIM toàn bộ test set)
# ================================================================

if [ "$TRAIN_ONLY" = false ] && [ "$DIAG_ONLY" = false ]; then

    echo ""
    echo "████████████████████████████████████████████████████████████"
    echo "  📊 PART 3 — PAPER SIMS (PSNR/SSIM)"
    echo "████████████████████████████████████████████████████████████"

    [ "$RUN_AWGN" = true ] && run_paper_sims "awgn" "AWGN" "" "" "AWGN"
    [ "$RUN_NOEQ" = true ] && run_paper_sims "noeq" "Rayleigh" "" "" "Rayleigh"
    [ "$RUN_EQ" = true ]    && run_paper_sims "eq" "Rayleigh" "_eq" "--rayleigh_equalize" "Rayleigh"

    echo ""
    echo "✅ PAPER SIMS XONG"
    [ "$RUN_AWGN" = true ] && echo "  📂 paper_sims_awgn/"
    [ "$RUN_NOEQ" = true ] && echo "  📂 paper_sims_noeq/"
    [ "$RUN_EQ" = true ]    && echo "  📂 paper_sims_eq/"
fi

echo ""

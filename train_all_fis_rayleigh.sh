#!/bin/bash
#=============================================================================
#  train_all_fis_rayleigh.sh
#  Huấn luyện TẤT CẢ mode FIS cho kênh Rayleigh (no-eq + eq)
#  Dùng: bash train_all_fis_rayleigh.sh [--noeq] [--eq] [--fast] [--dry]
#
#  --noeq   : chỉ chạy Rayleigh no-equalize (mặc định: chạy cả hai)
#  --eq     : chỉ chạy Rayleigh equalize   (mặc định: chạy cả hai)
#  --fast   : chế độ nhanh (ít epoch, chỉ 3 SNR: 1 7 13)
#  --dry    : chỉ in lệnh, không chạy thực sự
#=============================================================================

set -uo pipefail

# --------------- CẤU HÌNH ---------------
DATASET="cifar10"
IMAGE_SIZE=32
SNR_MIN=1
SNR_MAX=13
TRAIN_SNR="1 4 7 10 13"
EVAL_SNR="1 4 7 10 13"
BUDGET=1.0
BASE_DIR="exp_ctx"

# Chế độ fast (cho test nhanh)
if [[ " $@ " == *" --fast "* ]]; then
    echo "⚡ CHẾ ĐỘ FAST: chỉ 3 SNR (1, 7, 13)"
    TRAIN_SNR="1 7 13"
    EVAL_SNR="1 7 13"
fi

DRY_RUN=false
if [[ " $@ " == *" --dry "* ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN — chỉ in lệnh, không thực thi"
fi

RUN_NOEQ=true
RUN_EQ=true
if [[ " $@ " == *" --noeq "* ]] && [[ " $@ " != *" --eq "* ]]; then
    RUN_EQ=false
fi
if [[ " $@ " == *" --eq "* ]] && [[ " $@ " != *" --noeq "* ]]; then
    RUN_NOEQ=false
fi

# --------------- HÀM CHẠY ---------------
run_train() {
    local MODE=$1
    local CHANNEL_TAG=$2    # "Rayleigh" or "Rayleigh_eq"
    local SAVE_TAG=$3       # "noeq" or "eq"
    local EXTRA_ARGS="${4:-}"

    local SAVE_DIR="${BASE_DIR}/ckpts_${SAVE_TAG}_${MODE}_${CHANNEL_TAG}"

    local CMD="python train_fis_power.py \
  --dataset ${DATASET} \
  --image_size ${IMAGE_SIZE} \
  --channel Rayleigh \
  --mode ${MODE} \
  --budget ${BUDGET} \
  --snr_min ${SNR_MIN} \
  --snr_max ${SNR_MAX} \
  --train_snr_list ${TRAIN_SNR} \
  --eval_snr_list ${EVAL_SNR} \
  --save_dir ${SAVE_DIR} ${EXTRA_ARGS}"

    echo ""
    echo "============================================================"
    echo "  🚀 [${SAVE_TAG}] Mode: ${MODE} | Channel: Rayleigh"
    echo "  📁 Save: ${SAVE_DIR}/"
    echo "============================================================"
    echo "$CMD"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
    fi
}

# --------------- DANH SÁCH MODES ---------------
MODES=("linear" "importance_only" "snr_only" "full")
MODE_NAMES=("Linear" "Importance-Only" "SNR-Only" "Full")

TOTAL=0
DONE=0

# Đếm tổng số job
for mode in "${MODES[@]}"; do
    [ "$RUN_NOEQ" = true ] && TOTAL=$((TOTAL + 1))
    [ "$RUN_EQ" = true ]    && TOTAL=$((TOTAL + 1))
done

# --------------- RAYLEIGH NO-EQUALIZE ---------------
if [ "$RUN_NOEQ" = true ]; then
    echo ""
    echo "████████████████████████████████████████████████████████████"
    echo "  RAYLEIGH — NO EQUALIZE"
    echo "████████████████████████████████████████████████████████████"

    for i in "${!MODES[@]}"; do
        mode="${MODES[$i]}"
        name="${MODE_NAMES[$i]}"
        DONE=$((DONE + 1))
        echo ""
        echo "  ▶ [${DONE}/${TOTAL}] ${name} ..."
        run_train "$mode" "Rayleigh" "noeq" ""
    done
fi

# --------------- RAYLEIGH EQUALIZE ---------------
if [ "$RUN_EQ" = true ]; then
    echo ""
    echo "████████████████████████████████████████████████████████████"
    echo "  RAYLEIGH — WITH EQUALIZE"
    echo "████████████████████████████████████████████████████████████"

    for i in "${!MODES[@]}"; do
        mode="${MODES[$i]}"
        name="${MODE_NAMES[$i]}"
        DONE=$((DONE + 1))
        echo ""
        echo "  ▶ [${DONE}/${TOTAL}] ${name} ..."
        run_train "$mode" "Rayleigh_eq" "eq" "--rayleigh_equalize"
    done
fi

# --------------- TÓM TẮT ---------------
echo ""
echo "████████████████████████████████████████████████████████████"
echo "  ✅ HOÀN TẤT TẤT CẢ TRAINING"
echo "████████████████████████████████████████████████████████████"
echo ""
echo "  Checkpoint đã lưu:"
echo ""

if [ "$RUN_NOEQ" = true ]; then
    echo "  📂 Rayleigh No-EQ:"
    for mode in "${MODES[@]}"; do
        echo "     • ${BASE_DIR}/ckpts_noeq_${mode}_Rayleigh/fis_power_best.pth"
    done
fi

if [ "$RUN_EQ" = true ]; then
    echo "  📂 Rayleigh EQ:"
    for mode in "${MODES[@]}"; do
        echo "     • ${BASE_DIR}/ckpts_eq_${mode}_Rayleigh_eq/fis_power_best.pth"
    done
fi

echo ""
echo "  Sau đó chạy diagnose:"
echo ""
echo "  # No-EQ"
echo "  python diagnose_controller.py \\"
echo "    --linear_ckpt ${BASE_DIR}/ckpts_noeq_linear_Rayleigh/fis_power_best.pth \\"
echo "    --importance_only_ckpt ${BASE_DIR}/ckpts_noeq_importance_only_Rayleigh/fis_power_best.pth \\"
echo "    --snr_only_ckpt ${BASE_DIR}/ckpts_noeq_snr_only_Rayleigh/fis_power_best.pth \\"
echo "    --full_ckpt ${BASE_DIR}/ckpts_noeq_full_Rayleigh/fis_power_best.pth \\"
echo "    --channel Rayleigh --snr_db 13 \\"
echo "    --save_dir diag_noeq_rayleigh_snr13"
echo ""
echo "  # EQ"
echo "  python diagnose_controller.py \\"
echo "    --linear_ckpt ${BASE_DIR}/ckpts_eq_linear_Rayleigh_eq/fis_power_best.pth \\"
echo "    --importance_only_ckpt ${BASE_DIR}/ckpts_eq_importance_only_Rayleigh_eq/fis_power_best.pth \\"
echo "    --snr_only_ckpt ${BASE_DIR}/ckpts_eq_snr_only_Rayleigh_eq/fis_power_best.pth \\"
echo "    --full_ckpt ${BASE_DIR}/ckpts_eq_full_Rayleigh_eq/fis_power_best.pth \\"
echo "    --channel Rayleigh --snr_db 13 --rayleigh_equalize \\"
echo "    --save_dir diag_eq_rayleigh_snr13"
echo ""
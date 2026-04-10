#!/bin/bash
#=============================================================================
#  run_diag_rayleigh.sh
#  Chạy diagnose_controller.py cho Rayleigh No-EQ + EQ
#  Dùng: bash run_diag_rayleigh.sh [--noeq] [--eq] [--snr 13] [--dry]
#
#  --noeq   : chỉ chạy Rayleigh no-equalize (mặc định: chạy cả hai)
#  --eq     : chỉ chạy Rayleigh equalize   (mặc định: chạy cả hai)
#  --snr N  : SNR dB để đánh giá (mặc định: 13)
#  --dry    : chỉ in lệnh, không chạy thực sự
#=============================================================================

set -uo pipefail

BASE_DIR="exp_ctx"
SNR_DB=13
DRY_RUN=false
RUN_NOEQ=true
RUN_EQ=true

# --------------- Parse args ---------------
for arg in "$@"; do
    case $arg in
        --noeq)         RUN_EQ=false ;;
        --eq)           RUN_NOEQ=false ;;
        --snr)          shift; SNR_DB="${1:-13}" ;;
        --dry)          DRY_RUN=true ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN — chỉ in lệnh, không thực thi"
fi

# --------------- Hàm diagnose ---------------
run_diag() {
    local TAG=$1       # "noeq" hoặc "eq"
    local EXTRA="${2:-}"

    local SAVE_DIR="diag_${TAG}_rayleigh_snr${SNR_DB}"

    local CMD="python diagnose_controller.py \
  --linear_ckpt ${BASE_DIR}/ckpts_${TAG}_linear_Rayleigh${EXTRA:+_}${EXTRA}/fis_power_best.pth \
  --importance_only_ckpt ${BASE_DIR}/ckpts_${TAG}_importance_only_Rayleigh${EXTRA:+_}${EXTRA}/fis_power_best.pth \
  --snr_only_ckpt ${BASE_DIR}/ckpts_${TAG}_snr_only_Rayleigh${EXTRA:+_}${EXTRA}/fis_power_best.pth \
  --full_ckpt ${BASE_DIR}/ckpts_${TAG}_full_Rayleigh${EXTRA:+_}${EXTRA}/fis_power_best.pth \
  --channel Rayleigh --snr_db ${SNR_DB} ${EXTRA:+$EXTRA} \
  --save_dir ${SAVE_DIR}"

    echo ""
    echo "============================================================"
    echo "  🔬 DIAGNOSE: [${TAG}] Rayleigh | SNR = ${SNR_DB} dB"
    echo "  📁 Save: ${SAVE_DIR}/"
    echo "============================================================"
    echo "$CMD"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        eval "$CMD"
    fi
}

# --------------- No-EQ ---------------
if [ "$RUN_NOEQ" = true ]; then
    run_diag "noeq" ""
fi

# --------------- EQ ---------------
if [ "$RUN_EQ" = true ]; then
    run_diag "eq" "--rayleigh_equalize"
fi

echo ""
echo "✅ Xong! Kết quả diagnose:"
[ "$RUN_NOEQ" = true ] && echo "  📂 diag_noeq_rayleigh_snr${SNR_DB}/"
[ "$RUN_EQ" = true ]    && echo "  📂 diag_eq_rayleigh_snr${SNR_DB}/"
echo ""

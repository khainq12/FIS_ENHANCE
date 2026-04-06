#!/bin/bash
# =============================================================
# run_diag_budget.sh — Chạy diag_budget.py cho tất cả channels
#
# Cách dùng:
#   # Chạy tất cả channels (cần có sẵn checkpoint cho mỗi channel):
#   bash run_diag_budget.sh --all
#
#   # Chạy riêng 1 channel:
#   bash run_diag_budget.sh --channel AWGN
#   bash run_diag_budget.sh --channel Rayleigh --equalize
#   bash run_diag_budget.sh --channel Rician
#
#   # Chỉ chạy budget sweep (nhanh, skip heatmaps):
#   bash run_diag_budget.sh --channel AWGN --fast
#
#   # Custom checkpoint path:
#   bash run_diag_budget.sh --channel AWGN --ckpt /path/to/fis_power_best.pth
#
# Tham số:
#   --all           Chạy AWGN, Rayleigh (noeq), Rayleigh (eq), Rician
#   --channel CH    Chạy riêng channel CH
#   --equalize      Bật Rayleigh equalization (chỉ có tác dụng với Rayleigh)
#   --fast          Bỏ heatmaps (nhanh hơn)
#   --ckpt PATH     Path trực tiếp đến checkpoint (ghi đè auto-detect)
#   --ratio R       Channel ratio (default: 0.1667 = 1/6)
#   --snrs S1 S2..  Danh sách SNR (default: 1 7 13)
#   --budgets B1..  Danh sách budget (default: 0.0 0.1 0.25 0.5 0.75 1.0)
#   --root DIR      Thư mục gốc chứa checkpoint
#                   (default: /media/data/students/nguyenquangkhai)
#
# Path pattern:
#   {ROOT}/ckpts_{eq_tag}full_{CHANNEL}/fis_power_best.pth
#
#   Ví dụ:
#     /media/data/students/nguyenquangkhai/ckpts_full_AWGN/fis_power_best.pth
#     /media/data/students/nguyenquangkhai/ckpts_noeq_full_Rayleigh/fis_power_best.pth
#     /media/data/students/nguyenquangkhai/ckpts_eq_full_Rayleigh/fis_power_best.pth
#     /media/data/students/nguyenquangkhai/ckpts_full_Rician/fis_power_best.pth
# =============================================================

set -euo pipefail

# --------------- defaults ---------------
RATIO="0.1667"
SNR_LIST="1 7 13"
BUDGET_LIST="0.0 0.1 0.25 0.5 0.75 1.0"
CKPT_ROOT="/media/data/students/nguyenquangkhai"
DATASET="cifar10"
IMAGE_SIZE=32
BATCH_SIZE=64
SAMPLE_IDX=0
FAST=false
RUN_ALL=false
CHANNEL=""
EQUALIZE=false
CUSTOM_CKPT=""

# --------------- parse args ---------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --equalize)
            EQUALIZE=true
            shift
            ;;
        --fast)
            FAST=true
            shift
            ;;
        --ckpt)
            CUSTOM_CKPT="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --snrs)
            shift
            SNR_LIST="$*"
            break
            ;;
        --budgets)
            shift
            BUDGET_LIST="$*"
            break
            ;;
        --root)
            CKPT_ROOT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --sample)
            SAMPLE_IDX="$2"
            shift 2
            ;;
        -h|--help)
            head -n 30 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# --------------- helper functions ---------------
resolve_ckpt() {
    local ch="$1"
    local eq_tag="$2"

    if [[ -n "$CUSTOM_CKPT" ]]; then
        echo "$CUSTOM_CKPT"
        return
    fi

    # Path pattern: {ROOT}/ckpts_{eq_tag}full_{CHANNEL}/fis_power_best.pth
    # Lưu ý: AWGN cũng nằm trong ckpts_noeq_full_AWGN/ (luôn có tag noeq_ nếu không eq)
    # Ví dụ:
    #   ckpts_noeq_full_AWGN/fis_power_best.pth
    #   ckpts_noeq_full_Rayleigh/fis_power_best.pth
    #   ckpts_eq_full_Rayleigh/fis_power_best.pth
    #   ckpts_full_Rician/fis_power_best.pth
    local candidates=(
        "${CKPT_ROOT}/ckpts_${eq_tag}full_${ch}/fis_power_best.pth"
        "${CKPT_ROOT}/ckpts_full_${ch}/fis_power_best.pth"
        "${CKPT_ROOT}/ckpts_noeq_full_${ch}/fis_power_best.pth"
        "${CKPT_ROOT}/ckpts_${eq_tag}${ch}/fis_power_best.pth"
        "${CKPT_ROOT}/ckpts_${ch}/fis_power_best.pth"
        "./ckpts_${eq_tag}full_${ch}/fis_power_best.pth"
        "./ckpts_full_${ch}/fis_power_best.pth"
        "./ckpts_noeq_full_${ch}/fis_power_best.pth"
    )

    for p in "${candidates[@]}"; do
        if [[ -f "$p" ]]; then
            echo "$p"
            return
        fi
    done

    # Fallback: tìm fis_power_best.pth gần nhất
    local found
    found=$(find . -name "fis_power_best.pth" -type f 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
        echo "$found"
        return
    fi

    echo ""
    return
}

run_diag() {
    local ch="$1"
    local eq_tag="$2"
    local eq_flag=""

    local save_dir="diag_budget_out_${eq_tag}${ch}"
    local ckpt
    ckpt=$(resolve_ckpt "$ch" "$eq_tag")

    if [[ -z "$ckpt" ]]; then
        echo "=========================================="
        echo "  SKIP: ${eq_tag}${ch} — không tìm thấy checkpoint"
        echo "  Đặt --ckpt PATH hoặc kiểm tra ${CKPT_ROOT}/"
        echo "=========================================="
        return
    fi

    echo "=========================================="
    echo "  Channel: ${eq_tag}${ch}"
    echo "  Checkpoint: ${ckpt}"
    echo "  Output: ${save_dir}/"
    echo "=========================================="

    local extra_args=()
    if [[ "$EQUALIZE" == true && "$ch" == "Rayleigh" ]]; then
        extra_args+=(--rayleigh_equalize)
    fi
    if [[ "$FAST" == true ]]; then
        extra_args+=(--no_heatmaps)
    fi

    # Convert SNR_LIST and BUDGET_LIST to space-separated for nargs+
    local snr_args=""
    for s in $SNR_LIST; do
        snr_args="$snr_args $s"
    done

    local budget_args=""
    for b in $BUDGET_LIST; do
        budget_args="$budget_args $b"
    done

    python diag_budget.py \
        --fis_ckpt "$ckpt" \
        --channel "$ch" \
        --ratio "$RATIO" \
        --dataset "$DATASET" \
        --image_size "$IMAGE_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --sample_index "$SAMPLE_IDX" \
        --snr_list $snr_args \
        --budget_list $budget_args \
        --save_dir "$save_dir" \
        "${extra_args[@]+"${extra_args[@]}"}"

    echo ""
    echo "  Done: ${save_dir}/"
    echo ""
}

# --------------- main ---------------
echo "============================================"
echo "  FIS Budget Diagnostic Tool"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

if [[ "$RUN_ALL" == true ]]; then
    echo ">>> Chạy tất cả channels..."
    echo ""

    # AWGN
    run_diag "AWGN" ""

    # Rayleigh — không equalize
    run_diag "Rayleigh" "noeq_"

    # Rayleigh — có equalize
    run_diag "Rayleigh" "eq_"

    # Rician
    run_diag "Rician" ""

    echo "============================================"
    echo "  TẤT CẢ XONG"
    echo "============================================"

elif [[ -n "$CHANNEL" ]]; then
    run_diag "$CHANNEL" ""
else
    echo "Cách dùng:"
    echo "  bash run_diag_budget.sh --all                              # chạy tất cả"
    echo "  bash run_diag_budget.sh --channel AWGN                     # riêng AWGN"
    echo "  bash run_diag_budget.sh --channel Rayleigh --equalize      # Rayleigh + eq"
    echo "  bash run_diag_budget.sh --channel AWGN --fast              # skip heatmaps"
    echo "  bash run_diag_budget.sh --channel AWGN --ckpt /path/to.pth # custom ckpt"
    echo ""
    echo "Chạy mặc định: AWGN"
    echo ""
    run_diag "AWGN" ""
fi

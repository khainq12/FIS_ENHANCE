#!/bin/bash
set -e

echo "========================================"
echo "RUN EXPLAIN (FIGURE 3 - ALL SETTINGS)"
echo "========================================"

# activate env
source /media/data/students/nguyenquangkhai/project1/jscc311/bin/activate

ROOT=/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch
DATA=/media/data/students/nguyenquangkhai

# 🔥 Channels
CHANNELS=("AWGN" "Rayleigh" "Rician")

# 🔥 Modes (chọn model bạn muốn visualize, thường dùng FULL)
MODE="full"

########################################
# 🔥 LOOP EQ vs NOEQ
########################################
for EQ in "noeq" "eq"; do

echo "========================================"
echo "SETTING: $EQ"
echo "========================================"

# flag cho rayleigh
RAY_FLAG=""
if [[ "$EQ" == "eq" ]]; then
  RAY_FLAG="--rayleigh_equalize"
fi

########################################
for CH in "${CHANNELS[@]}"; do

echo "----------------------------------------"
echo "CHANNEL: $CH ($EQ)"
echo "----------------------------------------"

# 🔥 FIX PATH CHECKPOINT
CKPT=$DATA/ckpts_${EQ}_${MODE}_${CH}/fis_power_best.pth

# check tồn tại
if [ ! -f "$CKPT" ]; then
    echo "❌ Missing checkpoint: $CKPT"
    continue
fi

# output folder
OUT_DIR=$DATA/figures_${EQ}_${CH}

mkdir -p $OUT_DIR

echo ">>> Running explain..."
echo "CKPT: $CKPT"
echo "OUT:  $OUT_DIR"

python $ROOT/test1.py \
    --channel $CH \
    --ckpt $CKPT \
    --save_dir $OUT_DIR \
    $RAY_FLAG

done
done

echo "========================================"
echo "✅ DONE ALL EXPLAIN"
echo "========================================"
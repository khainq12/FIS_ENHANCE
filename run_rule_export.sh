#!/bin/bash
set -e

echo "========================================"
echo "EXPORT RULE USAGE TABLES (ALL SETTINGS)"
echo "========================================"

# activate env
source /media/data/students/nguyenquangkhai/project1/jscc311/bin/activate

ROOT=/media/data/students/nguyenquangkhai
SCRIPT=/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/export_rule_table.py

OUT_DIR=$ROOT/tables_rule_usage
mkdir -p $OUT_DIR

# 🔥 Channels (đúng format bạn yêu cầu)
CHANNELS=("AWGN" "Rayleigh" "Rician")

########################################
# LOOP eq vs noeq
########################################
for EQ in "noeq" "eq"; do

echo "========================================"
echo "SETTING: $EQ"
echo "========================================"

for CH in "${CHANNELS[@]}"; do

echo "----------------------------------------"
echo "CHANNEL: $CH ($EQ)"
echo "----------------------------------------"

python $SCRIPT \
    --channel $CH \
    --root $ROOT \
    --output_dir $OUT_DIR \
    --eq_tag $EQ

done
done

echo "========================================"
echo "✅ DONE EXPORT ALL RULE TABLES"
echo "========================================"
#!/bin/bash
set -e

echo "========================================"
echo "Deep JSCC + FIS (CelebA 256 - CLEAN)"
echo "========================================"

source /media/data/students/nguyenquangkhai/project1/jscc311/bin/activate

ROOT=/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch
DATA=/media/data/students/nguyenquangkhai

# 📁 Dataset CelebA 256
CELEBA=/media/data/students/nguyenquangkhai/celeba256/face

CHANNELS=("AWGN" "Rayleigh" "Rician")
MODES=("linear" "importance_only" "snr_only" "full")

SNRS="1.0,4.0,7.0,10.0,13.0"

########################################
for CH in "${CHANNELS[@]}"; do

echo "========================================"
echo "CHANNEL: $CH"
echo "========================================"

########################################
# 1. TRAIN BASELINE
########################################
echo ">>> TRAIN BASELINE ($CH)"

python $ROOT/train_baseline.py \
  --dataset celebahq \
  --data_root $CELEBA \
  --image_size 256 \
  --channel $CH \
  --ratio 0.0833333333 \
  --epochs 100 \
  --batch_size 32 \
  --snr_min 0 --snr_max 20 \
  --save_dir $DATA/ckpts_baseline_${CH}_celeba

########################################
# 2. TRAIN + SIM từng MODE
########################################
for MODE in "${MODES[@]}"; do

echo "----------------------------------------"
echo ">>> MODE: $MODE ($CH)"
echo "----------------------------------------"

########################################
# TRAIN
########################################
python $ROOT/train_fis_power.py \
  --dataset celebahq \
  --data_root $CELEBA \
  --image_size 256 \
  --channel $CH \
  --mode $MODE \
  --budget 1.0 \
  --ratio 0.0833333333 \
  --epochs 100 \
  --batch_size 32 \
  --snr_min 0 --snr_max 20 \
  --save_dir $DATA/ckpts_${MODE}_${CH}_celeba

########################################
# SIMULATION
########################################
python $ROOT/run_paper_sims.py \
  --dataset celebahq \
  --data_root $CELEBA \
  --image_size 256 \
  --channel $CH \
  --baseline_ckpt $DATA/ckpts_baseline_${CH}_celeba/baseline_best.pth \
  --fis_ckpt $DATA/ckpts_${MODE}_${CH}_celeba/fis_power_best.pth \
  --modes baseline,$MODE \
  --snrs $SNRS \
  --budgets 1.0 \
  --batch_size 16 \
  --save_dir $DATA/paper_sims_${CH}_${MODE}_celeba

########################################
# CHECK FILE
########################################
JSON_FILE=$DATA/paper_sims_${CH}_${MODE}_celeba/paper_sims_results.json

if [ ! -f "$JSON_FILE" ]; then
  echo "❌ Missing JSON: $JSON_FILE"
  exit 1
fi

########################################
# TABLE
########################################
python $ROOT/make_tables_from_json.py \
  --json $JSON_FILE \
  --budget 1.0 \
  --methods baseline,$MODE \
  --snrs $SNRS \
  --out $DATA/tables_${CH}_${MODE}_celeba.tex

done

########################################
# 3. MERGE ALL MODES
########################################
echo "========================================"
echo ">>> FINAL MERGE ($CH)"
echo "========================================"

OUT_JSON=$DATA/paper_sims_${CH}_ALL_celeba.json

python $ROOT/merge_json.py \
  $OUT_JSON \
  $DATA/paper_sims_${CH}_linear_celeba/paper_sims_results.json \
  $DATA/paper_sims_${CH}_importance_only_celeba/paper_sims_results.json \
  $DATA/paper_sims_${CH}_snr_only_celeba/paper_sims_results.json \
  $DATA/paper_sims_${CH}_full_celeba/paper_sims_results.json

########################################
# CHECK MERGED FILE
########################################
if [ ! -f "$OUT_JSON" ]; then
  echo "❌ Merge failed!"
  exit 1
fi

########################################
# FINAL TABLE
########################################
python $ROOT/make_tables_from_json.py \
  --json $OUT_JSON \
  --budget 1.0 \
  --methods baseline,linear,importance_only,snr_only,full \
  --snrs $SNRS \
  --out $DATA/tables_${CH}_ALL_celeba.tex

done

echo "========================================"
echo "✅ ALL DONE (CELEBA 256)"
echo "========================================"
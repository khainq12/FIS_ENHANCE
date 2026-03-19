#!/bin/bash
set -e

echo "========================================"
echo "Deep JSCC + FIS (FAIR EVAL - FIXED)"
echo "========================================"

source /media/data/students/nguyenquangkhai/project1/jscc311/bin/activate

ROOT=/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch
DATA=/media/data/students/nguyenquangkhai

CHANNELS=("AWGN" "Rayleigh" "Rician")
MODES=("linear" "importance_only" "snr_only" "full")

SNRS="1,4,7,10,13"

########################################
for CH in "${CHANNELS[@]}"; do

echo "========================================"
echo "CHANNEL: $CH"
echo "========================================"

########################################
# 1. TRAIN BASELINE (1 lần)
########################################
echo ">>> TRAIN BASELINE ($CH)"

python $ROOT/train_baseline.py \
  --dataset cifar10 \
  --image_size 32 \
  --channel $CH \
  --ratio 0.0833333333 \
  --epochs 5 \
  --batch_size 128 \
  --snr_min 0 --snr_max 20 \
  --save_dir $DATA/ckpts_baseline_${CH}

########################################
# 2. TRAIN ALL FIS
########################################
for MODE in "${MODES[@]}"; do

echo ">>> TRAIN FIS: $MODE ($CH)"

python $ROOT/train_fis_power.py \
  --dataset cifar10 \
  --image_size 32 \
  --channel $CH \
  --mode $MODE \
  --budget 1.0 \
  --ratio 0.0833333333 \
  --epochs 5 \
  --batch_size 128 \
  --snr_min 0 --snr_max 20 \
  --save_dir $DATA/ckpts_${MODE}_${CH}

done

########################################
# 3. SINGLE FAIR EVALUATION (QUAN TRỌNG)
########################################
echo ">>> RUN FAIR EVAL ($CH)"

python $ROOT/run_paper_sims.py \
  --dataset cifar10 \
  --image_size 32 \
  --channel $CH \
  --baseline_ckpt $DATA/ckpts_baseline_${CH}/baseline_best.pth \
  --fis_ckpt_root $DATA \
  --modes baseline,linear,importance_only,snr_only,full \
  --snrs $SNRS \
  --budgets 1.0 \
  --save_dir $DATA/paper_sims_${CH}_ALL

########################################
# 4. TABLE
########################################
python $ROOT/make_tables_from_json.py \
  --json $DATA/paper_sims_${CH}_ALL/paper_sims_results.json \
  --budget 1.0 \
  --methods baseline,linear,importance_only,snr_only,full \
  --snrs $SNRS \
  --out $DATA/tables_${CH}_ALL.tex

done

echo "========================================"
echo "✅ DONE - FAIR COMPARISON GUARANTEED"
echo "========================================"
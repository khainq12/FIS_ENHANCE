#!/bin/bash
set -e

echo "========================================"
echo "Deep JSCC + FIS (BUDGET SWEEP VERSION)"
echo "========================================"

source /media/data/students/nguyenquangkhai/project1/jscc311/bin/activate

ROOT=/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch
DATA=/media/data/students/nguyenquangkhai

CHANNELS=("AWGN" "Rayleigh" "Rician")
MODES=("linear" "importance_only" "snr_only" "full")

SNRS="1,4,7,10,13"
BUDGETS=("0.25" "0.5" "0.75" "1.0")

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
  --epochs 100 \
  --batch_size 128 \
  --snr_min 0 --snr_max 20 \
  --save_dir $DATA/ckpts_baseline_${CH}

########################################
# 2. TRAIN ALL FIS (1 lần)
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
  --epochs 100 \
  --batch_size 128 \
  --snr_min 0 --snr_max 20 \
  --save_dir $DATA/ckpts_${MODE}_${CH}

done

########################################
# 3. BUDGET SWEEP EVALUATION
########################################
echo ">>> RUN BUDGET SWEEP ($CH)"

for B in "${BUDGETS[@]}"; do

echo ">>> Budget = $B"

python $ROOT/run_paper_sims.py \
  --dataset cifar10 \
  --image_size 32 \
  --channel $CH \
  --baseline_ckpt $DATA/ckpts_baseline_${CH}/baseline_best.pth \
  --fis_ckpt_root $DATA \
  --modes baseline,linear,importance_only,snr_only,full \
  --snrs $SNRS \
  --budgets $B \
  --save_dir $DATA/paper_sims_${CH}_B${B}

done

########################################
# 4. MERGE JSON → CSV
########################################
echo ">>> MERGING JSON TO CSV ($CH)"

python $ROOT/merge_json.py \
  --input_dir $DATA \
  --pattern "paper_sims_${CH}_B*/paper_sims_results.json" \
  --output $DATA/metrics_cifar10_${CH,,}_budget_sweep.csv

########################################
# 5. METADATA (NHANH GỌN)
########################################
echo ">>> SAVING METADATA ($CH)"

cat <<EOL > $DATA/exp_metadata_cifar10_${CH,,}.json
{
  "dataset": "cifar10",
  "channel": "$CH",
  "snrs": [$SNRS],
  "budgets": [0.25,0.5,0.75,1.0],
  "batch_size": 128,
  "epochs": 100
}
EOL

done

echo "========================================"
echo "✅ DONE - READY FOR PAPER"
echo "========================================"
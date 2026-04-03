#!/bin/bash
set -e

echo "========================================"
echo "Deep JSCC + FIS (CELEBA-HQ 256)"
echo "========================================"

# activate env
source /media/data/students/nguyenquangkhai/project1/jscc311/bin/activate

ROOT=/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch
DATA=/media/data/students/nguyenquangkhai
DTS=/media/data/students/nguyenquangkhai/celeba256/face

CHANNELS=("AWGN" "Rayleigh" "Rician")
MODES=("linear" "importance_only" "snr_only" "full")

SNRS="1,4,7,10,13"
BUDGETS=("0.25" "0.5" "0.75" "1.0")

########################################
for EQ in "noeq" "eq"; do

echo "========================================"
echo "SETTING: $EQ"
echo "========================================"

RAY_FLAG=""
if [[ "$EQ" == "eq" ]]; then
  RAY_FLAG="--rayleigh_equalize"
fi

########################################
for CH in "${CHANNELS[@]}"; do

echo "========================================"
echo "CHANNEL: $CH ($EQ)"
echo "========================================"

########################################
# 1. TRAIN BASELINE
########################################
echo ">>> TRAIN BASELINE ($CH)"

python $ROOT/train_baseline.py \
  --dataset celebahq \
  --data_root $DTS \
  --image_size 256 \
  --channel $CH \
  --ratio 0.17 \
  --epochs 100 \
  --batch_size 16 \
  --num_workers 4 \
  --snr_min 0 --snr_max 20 \
  ${RAY_FLAG:+$RAY_FLAG} \
  --save_dir $DATA/ckpts_${EQ}_baseline_${CH}

########################################
# 2. TRAIN FIS
########################################
for MODE in "${MODES[@]}"; do

echo ">>> TRAIN FIS: $MODE ($CH)"

python $ROOT/train_fis_power.py \
  --dataset celebahq \
  --data_root $DTS \
  --image_size 256 \
  --channel $CH \
  --mode $MODE \
  --budget 1.0 \
  --ratio 0.17 \
  --epochs 100 \
  --batch_size 16 \
  --num_workers 4 \
  --snr_min 0 --snr_max 20 \
  ${RAY_FLAG:+$RAY_FLAG} \
  --save_dir $DATA/ckpts_${EQ}_${MODE}_${CH}

done

########################################
# 3. BUDGET SWEEP
########################################
echo ">>> RUN BUDGET SWEEP ($CH)"

for B in "${BUDGETS[@]}"; do

echo ">>> Budget = $B"

python $ROOT/run_paper_sims.py \
  --dataset celebahq \
  --data_root $DTS \
  --image_size 256 \
  --channel $CH \
  --baseline_ckpt $DATA/ckpts_${EQ}_baseline_${CH}/baseline_best.pth \
  --fis_ckpt_root $DATA \
  --eq_tag $EQ \
  ${RAY_FLAG:+$RAY_FLAG} \
  --modes baseline,linear,importance_only,snr_only,full \
  --snrs $SNRS \
  --budgets $B \
  --save_dir $DATA/paper_sims_${EQ}_${CH}_B${B}

done

########################################
# 4. MERGE JSON -> CSV
########################################
echo ">>> MERGING JSON TO CSV ($CH)"

python $ROOT/merge_json.py \
  --input_dir $DATA \
  --pattern "paper_sims_${EQ}_${CH}_B*/paper_sims_results.json" \
  --output $DATA/metrics_celeba_${EQ}_${CH,,}_budget_sweep.csv

########################################
# 5. SAVE METADATA
########################################
echo ">>> SAVING METADATA ($CH)"

cat <<EOL > $DATA/exp_metadata_celeba_${EQ}_${CH,,}.json
{
  "dataset": "celebahq",
  "data_root": "$DTS",
  "channel": "$CH",
  "setting": "$EQ",
  "snrs": [$SNRS],
  "budgets": [0.25,0.5,0.75,1.0],
  "batch_size": 16,
  "epochs": 100,
  "rayleigh_equalize": $([[ "$EQ" == "eq" ]] && echo true || echo false)
}
EOL

done
done

echo "========================================"
echo "✅ DONE - CELEBA-HQ READY"
echo "========================================"
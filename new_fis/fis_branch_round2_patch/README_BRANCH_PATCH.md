
# FIS_ENHANCE branch-ready patch (round 2)

This package is intended for a **new experimental branch** so that the current paper branch remains untouched.

## Recommended branch

```bash
git checkout main
git pull
git tag -a paper-fis-v1-stable -m "Stable version used for current manuscript"
git checkout -b feature/fis-channel-aware-controller-r2
```

## What is included

### Core controller/channel patches
- `patches/channel.py`
- `patches/fis_modules.py`
- `patches/model.py`
- `patches/train_baseline.py`

These add a channel-aware descriptor (`gamma_eff_norm`) to the full FIS while keeping the backbone unchanged.

### Optional experiment tools
- `optional_tools/run_paper_sims.py`
- `optional_tools/diagnose_controller.py`
- `optional_tools/make_tables_from_json.py`
- `optional_tools/export_rule_table.py`

These patches add:
- paired evaluation with shared fading context
- logging of `gamma_eff_db`, `gamma_eff_norm`, `h_abs2`, `posteq_noise_var`, `eq_quality`
- diagnostics that correlate `A` with channel reliability
- existing table export convenience from the previous patch

## Apply patches

```bash
cp patches/channel.py .
cp patches/fis_modules.py .
cp patches/model.py .
cp patches/train_baseline.py .
cp optional_tools/run_paper_sims.py .
cp optional_tools/diagnose_controller.py .
cp optional_tools/make_tables_from_json.py .
cp optional_tools/export_rule_table.py .
```

## First experiments to run

Run only the following sequence first:
1. baseline (unchanged architecture)
2. linear
3. importance_only
4. snr_only
5. full **with channel-aware context**

Prioritize:
- Rayleigh noeq
- Rayleigh eq
- Rician low-SNR

## Why this round-2 patch matters

The original full FIS used only content-derived importance and nominal SNR. That makes it **content-aware + SNR-aware**, but not truly **fading-aware**. This patch lets the full controller see an instantaneous channel-reliability descriptor while preserving the same average transmit-power budget.

The expected outcome is not guaranteed dominance on every channel, but a clearer separation of `full` from `linear` and `snr_only` under Rayleigh/Rician, especially at low SNR.

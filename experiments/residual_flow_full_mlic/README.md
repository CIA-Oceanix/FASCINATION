# Residual Flow Full-MLIC Experiment

## Goal
Build a full-depth residual enhancement experiment for MLIC++ with 157 channels (depth levels), starting from the reference implementation in `FASCINATION/experiments/residual_flow_chapron`.

## Why this branch is different from `residual_flow_chapron`
- `residual_flow_chapron` was developed around an RGB-like setup (3 channels) for paper-oriented experiments.
- This project targets full SSP-like tensors with 157 vertical levels.
- The frozen background codec remains MLIC++, but all data flow, reconstruction checks, and residual models must now be validated for high-channel tensors.

## Working principle
1. Train or load a frozen full-depth MLIC++ checkpoint (`in_channels=157`).
- Loading frozen models exemple are found in /Odyssey/private/o23gauvr/code/FASCINATION/src/test_metrics.py 
2. Decode `x_tilde` from the codec.
3. Learn residuals `r = x - x_tilde` with deterministic and/or stochastic residual models.
4. Reconstruct `x_hat = x_tilde + r_hat`.
5. Evaluate with both pointwise and structure-aware metrics.

## Document map
- `01_bootstrap_plan.md`: step-by-step implementation plan.
- `02_metrics_gap_vs_test_metrics.md`: metric comparison and missing-metric recommendations.

## Initial implementation scope

- Phase 1: strict reproducibility of baseline frozen-MLIC++ reconstruction on full-depth data.
- Phase 2: deterministic residual U-Net on 157 channels.
- Phase 3: flow/diffusion residual variants.
- Phase 4: extended evaluation and manuscript-ready reporting tables.

Important: Implement one phase per one phase

## Naming convention
- Experiment id: `residual_flow_full_mlic`
- Output root (recommended): `FASCINATION/experiments/residual_flow_full_mlic/outputs/`
- Logs root (recommended): `FASCINATION/experiments/residual_flow_full_mlic/logs/`

## Phase 3 implementation (flow only)

Flow-only training entrypoint:
- `FASCINATION/experiments/residual_flow_full_mlic/src/train_phase3_flow_only.py`

Default choices implemented for Phase 3:
- Residual target: train on scaled residual `s_t` (from Phase 2 outputs).
- Condition normalization: per-depth z-score on `x_tilde` (fit on train split only).
- Optimizer/schedule: `AdamW` + warmup cosine decay with LR floor.
- Checkpoints: always save `last_checkpoint.pth.tar` and metric-specific best checkpoints.

Expected Phase 2 arrays:
- `residual_target_scaled.npy`
- `residual_scale_per_depth.npy`
- `ae_unnorm.npy`

Example command:

```bash
python FASCINATION/experiments/residual_flow_full_mlic/src/train_phase3_flow_only.py \
	--residual-scaled-npy /path/to/residual_target_scaled.npy \
	--residual-scale-per-depth-npy /path/to/residual_scale_per_depth.npy \
	--ae-npy /path/to/ae_unnorm.npy \
	--output-dir /Odyssey/private/o23gauvr/code/FASCINATION/experiments/residual_flow_full_mlic/outputs \
	--run-name flow_only_std_scaled \
	--epochs 120 \
	--batch-size 4
```

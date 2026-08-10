# Step-by-Step Bootstrap Plan (Full-Depth MLIC++ Residual Flow)

## Current Status

- Phase 0 implemented.
- Phase 1 implemented.
- Phase 2 implemented.
- Active work starts at Phase 3.

## Phase 0 - Freeze the reference and define invariants (Implemented)

### Objective
Lock down what must stay identical to the current validated pipelines before introducing residual models.

### Invariants
- Same data split policy as your current production experiments.
- Same normalization policy used for full-depth MLIC++ training/evaluation.
- Same frozen codec checkpoint logic during residual training.
- Residual model never changes transmitted bitrate (decoder-side correction only).

### Inputs

- Two python codes on which models are loaded in inference with pre treatment before computing metrics: norm/unormalization, check for uniform z axis ...
- /Odyssey/private/o23gauvr/code/FASCINATION/src/test_metrics.py

### Deliverables
- `invariants.md` (short checklist).
- One command line that runs codec-only inference and stores reconstruction metrics (RMSE, F1_score).

## Phase 1 - Full-depth codec baseline (157 channels) (Implemented)

### Objective
Confirm the base codec path is fully stable for `(B, 157, H, W)` tensors.

### Checks
1. Data loader emits exactly 157 channels.
2. MLIC++ config uses `in_channels=157` (and matching output channels).
3. Forward pass shape consistency: `x`, `x_tilde`, masks, and metadata.
4. Denormalization and depth-axis semantics are verified.

### Expected outputs
- Baseline metrics file (`codec_baseline_metrics.csv` or `.json`).
- A small debug artifact with shape and min/max statistics per batch.

## Phase 2 - Residual target definition and scaling (Implemented)

### Objective
Create robust residual targets for full-depth tensors.

### Steps
1. Compute residual target: `r = x - x_tilde`.
2. Compute residual scaling statistics (`std`, optional robust quantiles).
3. Validate residual distribution per depth level (to detect unstable levels).
4. Add optional output clamp only if normalization requires it.

### Expected outputs
- `residual_stats.json` with global and per-level summary stats.
- Residual sanity plots (mean/std vs depth level).

## Phase 3 - Residual Flow Only (No U-Net)

### Objective
Train and validate a pure residual flow model using Phase 2 residual targets (`r_t`) and scaled residuals (`s_t`) without adding deterministic U-Net mean residual.

### Steps
1. Train residual flow with frozen codec background (`x_tilde`) and Phase 2 residual supervision.
2. Keep evaluation protocol identical to baseline (same split/sampling/normalization).
3. Select the best checkpoint with a primary metric and at least one structure-aware metric.
4. Compare flow-only reconstruction against codec-only baseline.

### Expected outputs
- `flow/` checkpoints and logs.
- `flow_only_vs_codec.csv` including delta columns.

## Phase 4 - Corrected Residual Evaluation

### Objective
Evaluate corrected residual reconstructions produced by Phase 3 and quantify where correction helps or hurts.

### Steps
1. Reconstruct corrected fields from residual predictions.
2. Compute pointwise and structure-aware metrics with the production evaluation stack.
3. Analyze depth-wise behavior and failure cases.
4. Produce comparison tables versus codec baseline and raw residual-flow outputs.

### Expected outputs
- `corrected_residual_metrics.csv`.
- `corrected_residual_diagnostics.md`.

## Phase 5 - Add U-Net and Cascade (U-Net -> Flow)

### Objective
Introduce deterministic U-Net mean residual and evaluate cascade behavior relative to flow-only results.

### Steps
1. Train deterministic U-Net residual branch with frozen codec.
2. Train/evaluate cascade where flow models the post-U-Net corrected residual.
3. Sweep cascade mixing settings and checkpoint selection criteria.
4. Compare codec-only vs flow-only vs U-Net-only vs cascade.

### Expected outputs
- `unet/` checkpoints and logs.
- `cascade/` checkpoints and logs.
- `cascade_vs_flow_vs_unet.csv`.

## Phase 6 - Flow Initialization Ablation

### Objective
Evaluate initialization sensitivity for residual flow and cascade tracks.

### Steps
1. Run initialization ablations with:
	- Gaussian
	- scaled Gaussian
	- spatial high-pass
	- Fourier high-pass
	- Laplace
	- Rademacher
2. Compare each initialization on pointwise + structure-aware metrics.
3. Report best initialization per objective (RMSE vs extrema/fidelity trade-offs).
4. Keep training/evaluation protocol fixed across all initializations.

### Expected outputs
- `flow_initialization_ablation.csv`.
- `flow_initialization_summary.md`.

## Out of Scope (Current Plan)

- Diffusion experiments are deferred for now.
- Reason: diffusion does not appear in the current manuscript result set and is not required for the active milestone.

## Recommended execution order
1. Validate that Phase 0-2 artifacts are stable and reproducible.
2. Phase 3: residual flow only.
3. Phase 4: corrected residual evaluation.
4. Phase 5: add U-Net and cascade.
5. Phase 6: flow initialization ablation.

## Stop criteria per phase
- Do not move to next phase if shape/mask/normalization checks fail.
- Do not compare models before confirming same split and same evaluation sampling settings.
- Do not claim improvement from one metric only; require consistency with at least one structure-aware metric.

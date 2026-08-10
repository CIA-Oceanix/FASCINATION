# Metric Gap Analysis: Residual Stack vs Current test_metrics.py

## Scope compared
- Current production evaluation: `FASCINATION/src/test_metrics.py`
- Residual-flow experiment stack:
  - `FASCINATION/experiments/residual_flow_chapron/src/metrics.py`
  - `FASCINATION/experiments/residual_flow_chapron/src/metrics_minmax.py`
  - `FASCINATION/experiments/residual_flow_chapron/src/metrics_minmax_v2.py`

## Metrics already present in test_metrics.py
- Pointwise/global: RMSE, MAE, PSNR, R2, Pearson.
- Extremum/basic structure: ECS, extremum position error, basic extrema F1.
- Sequence/distribution: DTW, Wasserstein.
- Spectral: LSD, peak frequency error, power-spectrum Wasserstein.
- Perceptual: SSIM, MS-SSIM.
- Domain-specific: NSR (depth and spatial effective-resolution variants), RMSE_PCA.

## Relevant metrics available in residual stack but missing in test_metrics.py

### 1) Robust extrema detection variants
- `detect_extrema_v2` with side-support filtering and adaptive gradient thresholding.
- Why relevant: for 157-depth fields, naive turning-point detection is more sensitive to noise and local oscillations.

### 2) Weighted/aggregated extrema F1 variants
- Separate `min`, `max`, `both`, and combined F1 summaries.
- Weighted-by-depth variants (`*_weighted_h`).
- Why relevant: depth levels are not equally informative; weighted variants can better reflect physically meaningful strata.

### 3) Prominence-based extrema metrics
- Prominence-driven extrema detector and F1 (`f1_extrema_prominence`).
- Why relevant: penalizes missing physically prominent structures, not just any local sign change.

### 4) Prominence-weighted event matching diagnostics
- `prominence_matching_metrics` outputs weighted precision/recall/F1 plus:
  - localization MAE,
  - prominence relative MAE,
  - amplitude MAE,
  - count bias.
- Why relevant: provides error decomposition beyond a single F1 scalar.

### 5) Filtered-extrema metrics
- Extrema metrics after low-pass filtering (`lowpass_filter_torch_along_axis` + filtered F1 fields).
- Why relevant: distinguishes large-scale structure preservation from small-scale noise effects.

### 6) Extrema Wasserstein by subset
- Wasserstein on robust extrema, prominent extrema, and filtered extrema subsets.
- Why relevant: distribution-level mismatch on events can reveal failures hidden by RMSE.

### 7) Extrema-DTW family
- `compute_extrema_dtw_metrics` and associated min/max/both/combined summaries.
- Why relevant: compares ordered extremum sequences, not raw profile amplitude only.

### 8) Gradient-aware physical metrics
- `gradient_rmse` and `gradient_rmse_z`.
- Why relevant: gradient fidelity is important for ocean/acoustic diagnostics.

### 9) Per-level masked residual diagnostics
- `masked_mse_and_ratio_per_level`.
- Why relevant: identifies depth-localized failure modes, critical with 157 channels.

### 10) PSD shape score helper
- `psd_score_spatial`.
- Why relevant: complementary to LSD and peak-frequency metrics, captures spectral-shape agreement.

## Priority recommendation for your new experiment

### Priority A (add first)
1. Prominence-weighted event matching metrics.
2. Robust extrema detector v2 + min/max/both/combined F1.
3. Gradient RMSE (at least depth-axis variant).
4. Per-level masked MSE diagnostics.

### Priority B (add second)
1. Extrema-DTW family.
2. Extrema Wasserstein subsets (robust/prominent/filtered).
3. Filtered-extrema F1.

### Priority C (optional)
1. Full PSD-shape score if runtime budget allows.

## Practical integration strategy
1. Keep `test_metrics.py` as the stable backbone.
2. Import only selected residual metrics as a dedicated extension module.
3. Add a metric-group switch, for example:
   - `baseline` (current production set),
   - `extrema_plus` (priority A/B additions),
   - `full_residual_eval` (all enabled).
4. Write outputs in separate namespaces to avoid key collisions.

## Suggested output key prefixes
- `extrema_v2_*`
- `prom_*`
- `grad_*`
- `per_level_*`
- `extrema_dtw_*`

## Main conclusion
You already cover strong global and spectral metrics, but you are currently missing the most informative structure-aware diagnostics introduced in the residual-flow stack, especially prominence-aware extrema matching and gradient/per-level diagnostics. These are highly relevant for full-depth (157-level) residual enhancement.

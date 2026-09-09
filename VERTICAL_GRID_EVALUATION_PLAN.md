# Plan: fix the native-vs-uniform vertical grid inconsistency in evaluation

Status: **not started** — this document is a handoff plan, no code has been changed yet.
Owner: unassigned.
Context: written after an external code review flagged that the vertical geometry
used to compute reconstruction metrics is not consistent across the codebase,
and that the most recently active experiment line evaluates on a reinterpolated
grid rather than the native NATL60/eNATL60 vertical grid.

## 1. The problem, verified against the actual data

The raw inputs (`eNATL60_BLB002_sound_speed_regrid_0_botm.nc`,
`NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc`, both under `/Odyssey/public/.../celerity/`)
have `z = 300` native NEMO levels, **non-uniform spacing**: ~1.1 m near the
surface, widening to ~21 m by 2000 m depth, up to ~50 m by 6025 m at the
bottom. Both files share the exact same `z` array (checked with `diff` on
`ncdump -v z` output).

Every datamodule truncates to `z < 2000 m` via
`manage_nan="supress_with_max_depth"` (a pure boolean mask, no interpolation).
Counting directly: **exactly 157 of the 300 native levels satisfy `z < 2000`**.
That is where the "157" channel count throughout configs and pickle filenames
comes from — a subset of the native, non-uniform grid.

However, `src/autoencoder_datamodule_natl_enatl.py` has a `uniform_z` flag
(`__init__` default `False`, but the pickles actually referenced by current
experiments were built with it `True`). When true, `setup()` applies the
`z < 2000` mask *first* (still 157 native levels), then calls
`_uniform_depth()` (around line 602-609):

```python
z_uniform = np.linspace(float(da.z.min()), float(da.z.max()), len(da.z))
return da.interp(z=z_uniform)
```

This **replaces** the 157 native, non-uniform depths with 157 *evenly spaced*
depths spanning the same range. Measured directly from the real z-values:
`(1979.896 - 0.480) / 156 ≈ 12.69 m` uniform spacing.

The most recently active experiment line in the repo
(`experiments/latent_space_optimisation/...`, the newest work present, dated
Aug 2026) points `dm_path` at a pickle whose name includes `_filtered_z_uniform_`
(e.g. `enatl_natl_dm_157_196_256_norm_per_split_filtered_z_uniform_alternate_days_7_60_10.pkl`),
confirming `uniform_z=True` was used to build it. Meanwhile, the older/default
evaluation path (`src/compute_metrics.py`, `src/full_metrics.py`'s initial
`dm_path`, `src/pca_compo_dict.py`) defaults to a pickle **without**
`_z_uniform` in the name, i.e. the native grid.

**Net effect:** two different vertical geometries are in active use under the
identical "157-level" label, with no field, log line, or filename-independent
metadata recording which one produced a given metric. The switch is only
recoverable by tracing `dm_path` → filename convention → which datamodule
class built it → its `uniform_z` value.

### Why this is not just a labeling bug

Sound-speed profiles have their sharpest gradients (mixed-layer depth,
thermocline, surface duct — the features that matter most acoustically) in
the shallow part of the water column, exactly where the native grid is
finest (~1 m spacing). The uniform ~12.7 m regrid removes resolution
precisely where it is scientifically most important, and does so *before*
computing the "truth" against which reconstruction error is measured — so
reported RMSE/ECS on the `_z_uniform` pipeline are not measuring the same
thing as RMSE/ECS on the native-grid pipeline, and are likely biased
optimistic relative to the true native profile.

## 2. Fix options (pick one, or do them in order)

### Option 1 — Make the grid explicit (cheap, no retraining, do this first regardless)
Goal: stop new results from being silently ambiguous, and label existing ones.

- [ ] In `autoencoder_datamodule_natl_enatl.py` (and ideally all datamodule
      variants), persist `uniform_z` and the resulting `depth_array` explicitly
      onto whatever gets pickled/logged (not just `self.depth_array` in memory).
- [ ] In whichever script writes `run_config.json` (the
      `latent_space_optimisation` results pipeline, `src/latent_space_y_optimization*.py`),
      add `"vertical_grid": {"uniform_z": bool, "n_levels": int, "z_min": float, "z_max": float}`
      to the saved config.
- [ ] Write a one-off audit script (`src/audit_vertical_grid.py` or similar)
      that opens every `*.pkl` datamodule under `pickle/` and every
      `run_config.json` under `experiments/`, determines native-vs-uniform from
      the actual `depth_array` spacing (std of diffs / mean diff — see
      `compression_nsr_analysis.py`'s `z_uniform_check` for an existing
      near-identical heuristic to reuse), and writes a small CSV/report tagging
      each existing result. This retroactively resolves the ambiguity in
      `experiments/latent_space_optimisation/results_test_full_loss_gauss/*`
      without touching any numbers.
- [ ] Add a short paragraph to the repo's README (or a new
      `VERTICAL_GRID.md`) documenting the native-vs-uniform distinction, so
      future contributors don't reintroduce the ambiguity.

### Option 2 — Correct the reported evaluation metric without retraining
Goal: for models already trained on the uniform grid, report error against
the true native profile instead of the pre-smoothed one.

- [ ] In the metric-computation step used for `_z_uniform`-trained models
      (likely inside `src/compute_metrics.py` / `src/full_metrics.py` /
      wherever `latent_space_y_optimization.py` scores reconstructions),
      after producing the reconstruction on the uniform 157-point grid,
      interpolate it back onto the true native 157 depth values (available
      from the *non*-uniform pickle, or recomputable from the source `.nc`
      files with the same `z < 2000` mask) before computing RMSE / ECS depth
      error / any other per-depth metric.
- [ ] Compare against the real (never-interpolated) truth profile at those
      native depths — not against `ssp_truth_da.interp(z=z_uniform)` as
      `full_metrics.py` currently does at lines ~945-947.
- [ ] Re-run metric computation (not training) for the checkpoints listed in
      the latest `run_config.json` files under
      `experiments/latent_space_optimisation/`, and compare the corrected
      numbers against the currently reported ones to quantify how large the
      discrepancy actually is. This comparison is itself a useful sanity
      check before deciding whether Option 3 is worth the retraining cost.

### Option 3 — Full fix: evaluate (and likely retrain) on the native grid
Goal: remove the bias rather than just measuring it.

- [ ] Rebuild the datamodule pickle behind the active `latent_space_optimisation`
      experiments with `uniform_z=False`.
- [ ] Because the model consumes a fixed 157-channel input, no architecture
      change is required — but the *values* at each channel change (native vs.
      uniform depths), so this requires retraining, not just re-evaluation.
      Confirm this understanding before committing to the retraining cost.
- [ ] Retrain the checkpoint set referenced in the most recent
      `run_config.json` (`checkpoint_paths` list — ~24 checkpoints as of the
      Aug 2026 runs) on the native-grid pickle.
- [ ] Note this invalidates direct comparison with checkpoints already
      produced this year; decide whether old results should be archived
      separately or re-run.

## 3. Recommended sequencing

1. Do Option 1 unconditionally — it's cheap, low-risk, and required no
   matter which of Option 2/3 is chosen later.
2. Do Option 2 next and look at the size of the discrepancy on a few
   existing checkpoints. If the corrected metric is close to the reported
   one, Option 3 may not be worth the retraining cost. If it's large
   (plausible given the thermocline argument above), that's the evidence
   needed to justify Option 3.
3. Decide on Option 3 with whoever owns the `latent_space_optimisation`
   experiment line, since it affects in-flight results.

## 4. Open questions for whoever picks this up

- Is `uniform_z=True` intentional (e.g. a deliberate simplification for a
  specific downstream use, such as a fixed-rate codec that assumes uniform
  sampling), or was it introduced without realizing it changes the
  evaluation geometry? This changes whether Option 3 is a "fix" or a
  documented design choice that just needs Option 1.
- Should the native grid or the uniform grid be the project's standard
  going forward? If uniform sampling is required downstream (e.g. for
  `MLIC`/codec comparisons that assume a regular grid), consider whether a
  finer uniform grid (denser than 12.7 m, e.g. matching the native surface
  resolution) would recover the lost near-surface accuracy at the cost of
  more channels.

## 5. Evidence trail (for verification by whoever resumes this)

- Native grid dimensions/spacing: `ncdump -h` / `ncdump -v z` on
  `/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc`
  and the NATL60 equivalent.
- 157-level truncation: `manage_nan="supress_with_max_depth"`, `max_depth = 2000`,
  in every `autoencoder_datamodule*.py` (`_manage_nan_single_da` /
  equivalent method).
- `uniform_z` flag and reinterpolation: `src/autoencoder_datamodule_natl_enatl.py`,
  `__init__` (`uniform_z: bool = False`, line ~97), `_uniform_depth` (line
  ~602-609), applied in `setup()` (line ~638-640).
- Active experiment using the uniform-grid pickle: any
  `experiments/latent_space_optimisation/results_test_full_loss_gauss/*/run_config.json`,
  field `dm_path` containing `_filtered_z_uniform_`.
- Default (native-grid) pickle used elsewhere: `dm_path` defaults in
  `src/compute_metrics.py` (~line 1360), `src/full_metrics.py` (~line 809),
  `src/pca_compo_dict.py` (~line 158) — none contain `_z_uniform` in the
  filename.

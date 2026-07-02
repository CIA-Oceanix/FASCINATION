# Compression Model NSR (Noise-to-Signal Ratio) Analysis

## Overview

This module adapts the spectral resolution estimation method from `mod_spectral.py` (based on Ballarotta et al. 2019) to evaluate the effective spatial resolution of compression models.

The method computes the **Noise-to-Signal Ratio (NSR)** in the spectral domain:

$$\text{NSR}(\lambda) = \frac{PSD_{\text{error}}(\lambda)}{PSD_{\text{truth}}(\lambda)}$$

where:
- $PSD_{\text{error}}$ = Power Spectral Density of reconstruction error (truth - reconstructed)
- $PSD_{\text{truth}}$ = Power Spectral Density of original signal
- $\lambda$ = wavelength in spatial/depth domain

The **effective resolution** is defined as the wavelength where NSR reaches a target ratio (typically 0.5):

$$\lambda_{\text{eff}} = \text{wavelength where } NSR(\lambda) = 0.5$$

This means the error variance equals half the signal variance at frequencies above this resolution.

## Important: Non-Uniform Sampling Handling

### The Problem

The FFT-based spectral analysis (Welch's method) **requires uniform sampling**. If your data has **non-uniform/exponential depth sampling** (more points at surface, fewer at depth), the code **automatically detects and resamples** to prevent bias.

### How It Works

1. **Automatic Detection**: `check_sampling_uniformity()` quantifies spacing irregularity
   - Computes: uniformity = std(spacing) / mean(spacing)
   - Returns: True if uniformity < 5%, False otherwise

2. **Automatic Resampling**: If non-uniform detected and `auto_resample=True` (default):
   - Linear interpolation to uniform depth grid
   - Grid spacing = median of original spacing
   - Applied before spectral analysis

3. **Metadata Tracking**: NetCDF output includes:
   - `sampling_uniform`: bool - was original sampling uniform?
   - `sampling_uniformity`: float - irregularity metric
   - `sampling_note`: description of what was done

### Example

```python
# Exponential depth sampling: more points at surface
depth_array = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])  # NON-UNIFORM

results = compute_nsr_scores_ssp(
    truth_ssp, reconstructed_ssp, 
    depth_array=depth_array,
    auto_resample=True  # DEFAULT - handles non-uniform sampling
)

# Info logged:
# "Non-uniform sampling detected (uniformity=0.89)"
# "Auto-resampling to uniform grid for spectral analysis"
# "Uniform grid delta_x: 28.44 km"
```

### Why This Matters

| Issue | Impact | Solution |
|-------|--------|----------|
| FFT assumes uniform sampling | Spectral leakage, aliasing | Auto-interpolate to uniform grid |
| Wavenumber computation biased | Resolution wavelength wrong | Use correct fs from uniform grid |
| Multiple scales mixed | Low-frequency errors masked | Separate pre-analysis resampling |

## Rationale for Compression Models

Unlike the original method (which compares satellite altimetry maps with along-track observations), this adapted version:

1. **Compares original data with model reconstructions** instead of multiple independent sources
2. **Applies to any compression/reconstruction model** (autoencoders, MLIC++, etc.)
3. **Provides frequency-domain quality metrics** complementary to spatial metrics (RMSE, SSIM, etc.)
4. **Identifies the scale at which model errors dominate** the signal
5. **Handles non-uniform sampling automatically** (e.g., exponential depth grids)

## Key Files

### 1. `compression_nsr_analysis.py` (Main Module)

Core functions:

- **`check_sampling_uniformity()`** - Detect if sampling is uniform or non-uniform
  - Returns: (is_uniform: bool, uniformity: float)
  - Uniformity = std(spacing) / mean(spacing)
  
- **`resample_to_uniform_grid()`** - Manually resample non-uniform data to uniform grid
  - Supports 1D and 2D arrays
  - Uses linear/cubic/nearest interpolation
  
- **`compute_nsr_1d_profiles()`** - Compute NSR for 1D profiles with **automatic non-uniform handling**
  - Detects sampling uniformity if depth_array provided
  - Auto-resamples if non-uniform
  
- **`compute_nsr_2d_field()`** - Compute NSR for 2D spatial fields

- **`find_resolution_crossing()`** - Find wavelength where NSR crosses target threshold

- **`compute_nsr_scores_ssp()`** - Full pipeline for SSP data with NetCDF output
  - **RECOMMENDED**: Always pass depth_array for non-uniform handling
  
- **`compute_nsr_spatial_map()`** - Full pipeline for 2D spatial data

### 2. `nsr_integration_example.py` (Integration Examples)

Helper functions for integration with `full_metrics.py`:

- **`compute_compression_nsr_for_ssp()`** - Wrapper for SSP analysis
- **`compute_compression_nsr_for_2d_field()`** - Wrapper for 2D field analysis
- **`compute_nsr_profile_analysis()`** - Per-profile NSR computation
- **`integrate_nsr_with_full_metrics()`** - Direct integration with full metrics workflow

## Usage Examples

### Example 1: Compute NSR for SSP (Sound Speed Profile) Compression

```python
from compression_nsr_analysis import compute_nsr_scores_ssp
from pathlib import Path

# Assuming you have:
# - ssp_truth: original SSP data (shape: batch×depth)
# - ssp_reconstructed: reconstructed from your model
# - depth_array: depth coordinates (can be non-uniformly sampled!)

results = compute_nsr_scores_ssp(
    truth_ssp=ssp_truth,
    reconstructed_ssp=ssp_reconstructed,
    depth_array=depth_array,  # IMPORTANT: pass depth_array for non-uniform handling
    output_filename='nsr_ssp_results.nc',
    target_ratio=0.5,  # NSR threshold for resolution
    method_name='my_compression_model',
    auto_resample=True  # Automatically handles non-uniform sampling (default)
)

print(f"Effective Resolution: {results['resolution']:.4f} km")
print(f"Mean Coherence: {results['coherence'].mean():.4f}")
print(f"Sampling Info: {results['sampling_info']}")

# Log output will show if sampling was non-uniform:
# "Non-uniform sampling detected (uniformity=0.89). Resampling to uniform grid..."
```

### Example 2: Non-uniform depth sampling (exponential grid)

```python
# Typical oceanographic SSP: exponential depth sampling
depth = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
# More points at surface, sparse at depth - VERY non-uniform!

# The code automatically detects and resamples:
results = compute_nsr_scores_ssp(
    ssp_truth, ssp_ae, depth,
    output_filename='nsr_results.nc',
)

# Info logged:
# "Non-uniform sampling detected (uniformity=1.23)"
# "Auto-resampling to uniform grid for spectral analysis"
# "Uniform grid delta_x: 162.45 km"

# Results will be UNBIASED even though original sampling was non-uniform
# NetCDF output includes sampling metadata
```

### Example 2: Integrate with full_metrics.py workflow

```python
from pathlib import Path
import sys
sys.path.insert(0, '/Odyssey/private/o23gauvr/code/FASCINATION/src')

from nsr_integration_example import integrate_nsr_with_full_metrics

# After computing AE reconstructions in full_metrics.py:
nsr_results, profile_results = integrate_nsr_with_full_metrics(
    ssp_truth=ssp_truth,
    ssp_ae=ssp_ae,
    depth_array=depth_array,
    metric_path=Path('./metrics_output'),
    model_name='my_autoencoder',
    target_ratios=[0.25, 0.5, 0.75]
)

# Results include:
# - nsr_results[0.5]['resolution'] - effective resolution at NSR=0.5
# - profile_results['aggregate']['mean_resolution_km'] - average across profiles
# - profile_results['aggregate']['mean_coherence'] - frequency coherence
```

### Example 3: Compute NSR for 2D spatial fields

```python
from compression_nsr_analysis import compute_nsr_spatial_map

# For map projections or 2D field data (shape: time×lat×lon)
results = compute_nsr_spatial_map(
    truth_field=truth_maps,
    reconstructed_field=reconstructed_maps,
    delta_x=1.0,  # grid spacing in km (x-direction)
    delta_y=1.0,  # grid spacing in km (y-direction)
    grid_lat=lat_coords,
    grid_lon=lon_coords,
    output_filename='nsr_spatial_results.nc',
    target_ratio=0.5,
    method_name='compression_model'
)
```

### Example 4: Per-profile detailed analysis

```python
from nsr_integration_example import compute_nsr_profile_analysis

profile_results = compute_nsr_profile_analysis(
    truth_profiles=ssp_truth,      # shape: (n_profiles, n_depth)
    reconstructed_profiles=ssp_ae,
    depth_array=depth_array,
    method_name='my_model',
    target_ratio=0.5
)

# Get statistics
agg = profile_results['aggregate']
print(f"Mean resolution: {agg['mean_resolution_km']:.4f} ± {agg['std_resolution_km']:.4f} km")
print(f"Valid profiles: {agg['n_valid_profiles']}")
print(f"Mean RMSE: {agg['mean_rmse']:.6f}")
```

## Interpreting Results

### Output Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `wavenumber` | Spatial frequency | 1/km |
| `wavelength` | Wavelength (1/wavenumber) | km |
| `nsr` | Noise-to-Signal Ratio | dimensionless |
| `coherence` | Magnitude squared coherence | [0,1] |
| `resolution` | Effective resolution (NSR=0.5) | km |
| `psd_error` | Power spectral density of error | - |
| `psd_truth` | Power spectral density of original | - |

### Interpretation Guide

1. **Low NSR values** (< 0.1): Model captures these scales well, error is small relative to signal
2. **NSR ≈ 0.5** (target): Effective resolution threshold - error and signal power are balanced
3. **High NSR values** (> 1.0): Model cannot represent these scales, error dominates

4. **Resolution comparison**:
   - Larger resolution value = better performance (error at larger scales)
   - Compare across models: higher resolution = captures more detail

5. **Coherence**:
   - Values close to 1.0 indicate model captures phase relationships
   - Values close to 0 indicate significant phase errors

## Output Files

The functions generate NetCDF files with the following structure:

```
nsr_ssp_results.nc
├── Dimensions:
│   └── wavenumber: 128
├── Variables:
│   ├── wavenumber (f8, 1/km)
│   ├── wavelength (f8, km)
│   ├── nsr (f8, dimensionless)
│   ├── coherence (f8, [0,1])
│   ├── psd_error (f8)
│   ├── psd_truth (f8)
└── Global Attributes:
    ├── method: compression method name
    ├── target_nsr_ratio: 0.5
    ├── effective_resolution: wavelength in km
    └── resolution_definition: "Wavelength where NSR=0.5"
```

## Mathematical Details

### NSR Computation

Using Welch's method for robust PSD estimation:

1. **Segment data** into overlapping windows
2. **Apply Hanning window** to each segment
3. **Compute FFT** for each segment
4. **Average power spectra** across segments
5. **Compute ratio** for NSR

### Resolution Finding

Linear interpolation in log-space between crossing points:

$$\log(\lambda_{\text{eff}}) = \log(k_1) - \frac{\text{NSR}(k_1) - 0.5}{\text{NSR}(k_1) - \text{NSR}(k_2)} (\log(k_1) - \log(k_2))$$

where $k_1, k_2$ are wavenumbers bracketing the 0.5 crossing.

## Computational Parameters

Key tunable parameters:

```python
# Depth/spatial sampling
delta_x = None  # If None, computed automatically from depth_array
               # Only used for uniform grids; ignored if depth_array is non-uniform

# Spectral analysis window
nperseg = 128  # points per FFT window (larger = better frequency resolution)

# Resolution threshold  
target_ratio = 0.5  # NSR value for effective resolution

# Non-uniform sampling handling
auto_resample = True  # Auto-detect and fix non-uniform sampling (RECOMMENDED)
```

Recommendations:
- **Always pass `depth_array`** to enable automatic non-uniform sampling detection
- Use `auto_resample=True` (default) unless you have a specific reason not to
- Use `nperseg ≈ data_length / 8` for good balance
- Use `target_ratio = 0.5` for standard comparison (Ballarotta et al.)
- For conservative estimates: `target_ratio = 0.25`
- For optimistic estimates: `target_ratio = 0.75`

### Non-Uniform Sampling Control

```python
# Option 1: Auto-detect and auto-resample (RECOMMENDED)
results = compute_nsr_scores_ssp(
    truth_ssp, recon_ssp, depth_array,
    auto_resample=True  # Default
)

# Option 2: Detect but don't resample (gets warning if non-uniform)
results = compute_nsr_scores_ssp(
    truth_ssp, recon_ssp, depth_array,
    auto_resample=False  # Not recommended
)

# Option 3: Check sampling uniformity manually
from compression_nsr_analysis import check_sampling_uniformity

is_uniform, uniformity = check_sampling_uniformity(depth_array, tolerance=0.05)
print(f"Uniform: {is_uniform}, Irregularity: {uniformity:.3f}")

# Option 4: Resample manually before analysis
from compression_nsr_analysis import resample_to_uniform_grid

truth_resampled, uniform_depth = resample_to_uniform_grid(
    truth_ssp, depth_array, kind='linear'
)
recon_resampled, _ = resample_to_uniform_grid(
    reconstructed_ssp, depth_array, kind='linear'
)

results = compute_nsr_scores_ssp(
    truth_resampled, recon_resampled, uniform_depth,
    auto_resample=False  # Already uniform
)
```

## Bias Analysis: Non-Uniform Sampling

### Without Non-Uniform Handling (Biased)

```python
# BAD: Pass irregular depth but don't pass depth_array
wavenumber, nsr, coherence = compute_nsr_1d_profiles(
    truth_ssp, recon_ssp,
    delta_x=1.0  # Assumes uniform spacing - WRONG for exponential grid!
)
# Results biased - wavenumbers meaningless, spectral leakage

# Log: "No warning" - silently produces wrong results
```

### With Non-Uniform Handling (Correct)

```python
# GOOD: Pass depth_array for automatic handling
results = compute_nsr_scores_ssp(
    truth_ssp, recon_ssp, 
    depth_array=exponential_depth_array,
    auto_resample=True  # Detects and fixes non-uniformity
)
# Results correct - data resampled to uniform grid first

# Log output:
# "Non-uniform sampling detected (uniformity=0.95)"
# "Auto-resampling to uniform grid for spectral analysis"
# "Uniform grid delta_x: 28.44 km"
# Results stored with sampling metadata
```

## Integration with full_metrics.py

To add NSR metrics to your full_metrics.py workflow:

```python
# In full_metrics.py main computation section:
from FASCINATION.src.nsr_integration_example import integrate_nsr_with_full_metrics

# After computing AE reconstructions
if compute_spectral_metrics:
    nsr_results, profile_results = integrate_nsr_with_full_metrics(
        ssp_truth=ssp_truth,
        ssp_ae=ssp_ae,
        depth_array=depth_array,
        metric_path=plot_path,
        model_name=model_name,
        target_ratios=[0.25, 0.5, 0.75]
    )
    
    # Store results for plotting/analysis
    metrics['nsr'] = nsr_results
    metrics['profile_nsr'] = profile_results
```

## References

- **Ballarotta et al. (2019)**: "On the resolutions of ocean altimetry maps"
  Ocean Science, 15, 1091–1109
  https://doi.org/10.5194/os-15-1091-2019

- **Welch's Method**: Welch, P. (1967), "The use of fast Fourier transform for the
  estimation of power spectra: A method based on time averaging over short,
  modified periodograms"

## Troubleshooting

### Issue: All zeros in NSR output
**Cause**: Invalid or NaN values in input data
**Solution**: Check input arrays for NaN/Inf, use `np.isfinite()` to validate

### Issue: NSR never reaches 0.5
**Cause**: Error consistently above or below signal power
**Solution**: Check model reconstruction quality, may need different target_ratio

### Issue: Memory error with large arrays
**Cause**: Computing on full array at once
**Solution**: Process in batches, reshape to 1D for FFT

### Issue: Poor frequency resolution
**Cause**: `nperseg` too small
**Solution**: Increase `nperseg` or provide more data

## Authors

- Adapted from: mod_spectral.py (oceanographic spectral analysis)
- Adaptation for compression models: Based on Ballarotta et al. methodology
- Integration: FASCINATION project

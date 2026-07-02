"""
Example integration of compression_nsr_analysis with full_metrics.py

This script demonstrates how to use the NSR-based spectral resolution analysis
with compression models from full_metrics.py.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import scipy.signal
import sys

# Add path to FASCINATION modules
sys.path.insert(0, str(Path(__file__).parent))

from FASCINATION.src.compression_nsr_analysis import (
    compute_nsr_scores_ssp,
    compute_nsr_spatial_map,
    compute_nsr_1d_profiles,
    find_resolution_crossing,
)


def compute_compression_nsr_for_ssp(
    truth_ssp: np.ndarray,
    reconstructed_ssp: np.ndarray,
    depth_array: np.ndarray,
    output_dir: Path = None,
    target_ratio: float = 0.5,
    method_name: str = 'compression_model',
    auto_resample: bool = True,
) -> dict:
    """
    Compute NSR-based resolution metrics for SSP compression models.

    Parameters
    ----------
    truth_ssp : np.ndarray
        Original SSP data with shape (batch, depth) or (time, lat, lon, depth)
    reconstructed_ssp : np.ndarray
        Reconstructed SSP from model with same shape
    depth_array : np.ndarray
        Depth coordinate array. IMPORTANT: Can be non-uniformly sampled (e.g., exponential).
        Function automatically detects and handles non-uniform sampling.
    output_dir : Path, optional
        Directory to save NetCDF output
    target_ratio : float
        NSR threshold for resolution definition (default 0.5)
    method_name : str
        Name of compression method for metadata
    auto_resample : bool
        If True (default), automatically resample non-uniform depth grids to uniform 
        spacing before spectral analysis. Prevents bias.

    Returns
    -------
    dict
        Results containing wavenumber, wavelength, NSR, coherence, and resolution
    """
    
    # Create output directory if specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"nsr_analysis_{method_name}.nc"
    else:
        output_file = None
    
    # Compute NSR scores
    results = compute_nsr_scores_ssp(
        truth_ssp,
        reconstructed_ssp,
        depth_array=depth_array,
        output_filename=str(output_file) if output_file else None,
        target_ratio=target_ratio,
        method_name=method_name,
        auto_resample=auto_resample,
    )
    
    return results


def compute_compression_nsr_for_2d_field(
    truth_field: np.ndarray,
    reconstructed_field: np.ndarray,
    grid_lat: np.ndarray = None,
    grid_lon: np.ndarray = None,
    output_dir: Path = None,
    target_ratio: float = 0.5,
    method_name: str = 'compression_model',
    delta_x: float = 1.0,
    delta_y: float = 1.0,
) -> dict:
    """
    Compute NSR-based resolution metrics for 2D spatial field compression models.

    Parameters
    ----------
    truth_field : np.ndarray
        Original 2D field data with shape (time, lat, lon) or similar
    reconstructed_field : np.ndarray
        Reconstructed field from model with same shape
    grid_lat : np.ndarray, optional
        Latitude coordinates
    grid_lon : np.ndarray, optional
        Longitude coordinates
    output_dir : Path, optional
        Directory to save NetCDF output
    target_ratio : float
        NSR threshold for resolution definition (default 0.5)
    method_name : str
        Name of compression method
    delta_x : float
        Grid spacing in x-direction (km)
    delta_y : float
        Grid spacing in y-direction (km)

    Returns
    -------
    dict
        Results containing wavenumber, wavelength, NSR, coherence, and resolution
    """
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"nsr_analysis_2d_{method_name}.nc"
    else:
        output_file = None
    
    results = compute_nsr_spatial_map(
        truth_field,
        reconstructed_field,
        delta_x=delta_x,
        delta_y=delta_y,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        output_filename=str(output_file) if output_file else None,
        target_ratio=target_ratio,
        method_name=method_name,
    )
    
    return results


def compute_nsr_profile_analysis(
    truth_profiles: np.ndarray,
    reconstructed_profiles: np.ndarray,
    depth_array: np.ndarray,
    method_name: str = 'model',
    target_ratio: float = 0.5,
) -> dict:
    """
    Compute NSR for individual profiles and return detailed spectral analysis.

    Parameters
    ----------
    truth_profiles : np.ndarray
        Array of profiles with shape (n_profiles, n_depth)
    reconstructed_profiles : np.ndarray
        Reconstructed profiles with same shape
    depth_array : np.ndarray
        Depth values
    method_name : str
        Name of method for logging
    target_ratio : float
        NSR threshold

    Returns
    -------
    dict
        Per-profile and aggregate results
    """
    from scipy.signal import welch
    
    n_profiles = truth_profiles.shape[0]
    n_depth = truth_profiles.shape[1]
    
    resolutions = np.zeros(n_profiles)
    coherences = np.zeros(n_profiles)
    rmse_profiles = np.zeros(n_profiles)
    
    # Compute per-profile statistics
    for i in range(n_profiles):
        truth_prof = truth_profiles[i, :]
        recon_prof = reconstructed_profiles[i, :]
        
        # Skip if NaN or all zeros
        if np.all(np.isnan(truth_prof)) or np.sum(np.abs(truth_prof)) == 0:
            continue
        
        valid_mask = np.isfinite(truth_prof) & np.isfinite(recon_prof)
        truth_valid = truth_prof[valid_mask]
        recon_valid = recon_prof[valid_mask]
        
        if len(truth_valid) < 10:
            continue
        
        # RMSE
        rmse_profiles[i] = np.sqrt(np.mean((truth_valid - recon_valid)**2))
        
        # Coherence
        fs = 1.0 / np.mean(np.diff(depth_array))
        nperseg = min(64, len(truth_valid) // 2)
        try:
            _, coh = scipy.signal.coherence(
                truth_valid, recon_valid, fs=fs, nperseg=nperseg, noverlap=0
            )
            coherences[i] = np.mean(coh)
        except:
            coherences[i] = 0.0
        
        # NSR and resolution
        try:
            _, nsr, _ = compute_nsr_1d_profiles(
                truth_prof[np.newaxis, :],
                recon_prof[np.newaxis, :],
                delta_x=np.mean(np.diff(depth_array)),
                nperseg=nperseg,
            )
            wavenumber, _, _ = compute_nsr_1d_profiles(
                truth_prof[np.newaxis, :],
                recon_prof[np.newaxis, :],
                delta_x=np.mean(np.diff(depth_array)),
                nperseg=nperseg,
            )
            res, _ = find_resolution_crossing(wavenumber, nsr, target_ratio=target_ratio)
            resolutions[i] = res
        except:
            resolutions[i] = 0.0
    
    # Aggregate analysis across all profiles
    valid_res = resolutions[resolutions > 0]
    valid_coh = coherences[coherences > 0]
    
    aggregate_results = {
        'method': method_name,
        'target_nsr_ratio': target_ratio,
        'mean_resolution_km': np.mean(valid_res) if len(valid_res) > 0 else 0.0,
        'std_resolution_km': np.std(valid_res) if len(valid_res) > 1 else 0.0,
        'min_resolution_km': np.min(valid_res) if len(valid_res) > 0 else 0.0,
        'max_resolution_km': np.max(valid_res) if len(valid_res) > 0 else 0.0,
        'mean_coherence': np.mean(valid_coh) if len(valid_coh) > 0 else 0.0,
        'mean_rmse': np.mean(rmse_profiles[rmse_profiles > 0]),
        'n_valid_profiles': len(valid_res),
    }
    
    return {
        'aggregate': aggregate_results,
        'per_profile': {
            'resolutions': resolutions,
            'coherences': coherences,
            'rmse': rmse_profiles,
        }
    }


# ============================================================================
# Integration examples with full_metrics.py workflow
# ============================================================================

def integrate_nsr_with_full_metrics(
    ssp_truth: np.ndarray,
    ssp_ae: np.ndarray,
    depth_array: np.ndarray,
    metric_path: Path,
    model_name: str,
    target_ratios: list = [0.25, 0.5, 0.75],
):
    """
    Compute NSR metrics alongside existing full_metrics computations.

    Parameters
    ----------
    ssp_truth : np.ndarray
        Truth SSP data
    ssp_ae : np.ndarray
        Reconstructed SSP from model
    depth_array : np.ndarray
        Depth coordinates
    metric_path : Path
        Directory to save results
    model_name : str
        Name of compression model
    target_ratios : list
        NSR thresholds to evaluate at
    """
    
    metric_path = Path(metric_path)
    nsr_output_dir = metric_path / 'nsr_analysis'
    nsr_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Computing NSR-based Spectral Resolution for {model_name}")
    print(f"{'='*70}\n")
    
    # Compute for multiple target ratios
    nsr_results = {}
    for target_ratio in target_ratios:
        print(f"Computing NSR analysis with target_ratio = {target_ratio}...")
        
        results = compute_nsr_scores_ssp(
            ssp_truth,
            ssp_ae,
            depth_array,
            output_filename=str(nsr_output_dir / f"nsr_target_{target_ratio:.2f}_{model_name}.nc"),
            target_ratio=target_ratio,
            method_name=model_name,
        )
        
        nsr_results[target_ratio] = results
        
        print(f"  Effective resolution @ NSR={target_ratio}: {results['resolution']:.4f} km")
        print(f"  Mean coherence: {np.mean(results['coherence']):.4f}")
    
    # Profile-level analysis
    print(f"\nComputing per-profile NSR statistics...")
    profile_results = compute_nsr_profile_analysis(
        ssp_truth,
        ssp_ae,
        depth_array,
        method_name=model_name,
        target_ratio=0.5,
    )
    
    print(f"\n  Mean effective resolution: {profile_results['aggregate']['mean_resolution_km']:.4f} km")
    print(f"  Std resolution: {profile_results['aggregate']['std_resolution_km']:.4f} km")
    print(f"  Mean coherence: {profile_results['aggregate']['mean_coherence']:.4f}")
    print(f"  Mean RMSE: {profile_results['aggregate']['mean_rmse']:.4f}")
    
    # Save summary
    summary_file = nsr_output_dir / f"nsr_summary_{model_name}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"NSR Analysis Summary for {model_name}\n")
        f.write("="*70 + "\n\n")
        
        for target_ratio, results in nsr_results.items():
            f.write(f"Target NSR Ratio: {target_ratio}\n")
            f.write(f"  Effective Resolution: {results['resolution']:.4f} km\n")
            f.write(f"  Mean Coherence: {np.mean(results['coherence']):.4f}\n")
            f.write("\n")
        
        f.write("\nPer-Profile Statistics:\n")
        for key, val in profile_results['aggregate'].items():
            if isinstance(val, (int, float)):
                f.write(f"  {key}: {val:.6f}\n")
            else:
                f.write(f"  {key}: {val}\n")
    
    print(f"\nResults saved to {nsr_output_dir}")
    print(f"Summary written to {summary_file}\n")
    
    return nsr_results, profile_results


if __name__ == '__main__':
    print("Compression NSR Analysis - Example Usage")
    print("\nTo use with full_metrics.py:")
    print("  1. After computing AE reconstruction, call integrate_nsr_with_full_metrics()")
    print("  2. Or use compute_compression_nsr_for_ssp() for direct analysis")
    print("  3. Or use compute_nsr_spatial_map() for 2D field analysis")
    print("\nExample:")
    print("  results = integrate_nsr_with_full_metrics(")
    print("      ssp_truth, ssp_ae, depth_array,")
    print("      metric_path, model_name='my_model'")
    print("  )")

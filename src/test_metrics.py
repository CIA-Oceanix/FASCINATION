"""
Compute metrics for compression models - Cleaner modular version.

Supports checkpoint discovery, model loading, and selective metric computation.
"""

import os
import pickle
import sys
import torch
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import re

from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d, convolve
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed
from dtaidistance import dtw
import gc
import math
import struct
import subprocess
import shutil

running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0, running_path)
sys.path.insert(0, "/Odyssey/private/o23gauvr/code/FASCINATION")

from MLIC.MLIC.models import MLICPlusPlus
from MLIC.MLIC.utils.utils import Config
from FASCINATION.src.utils import unorm_ssp_arr_3D, norm_ssp_arr_3D, get_cfg_from_ckpt_path, load_model, getsize
from FASCINATION.src.compression_nsr_analysis import compute_nsr_along_depth, compute_nsr_spatial_map

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ============================================================================
# MEMORY LOGGING UTILITIES
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    return None

def log_data_shapes(data_dict, label=""):
    """Log shapes and estimated sizes of arrays in a dictionary."""
    log_str = f"\n{'='*70}\n{label}\n{'='*70}\n"
    for name, data in data_dict.items():
        if isinstance(data, (np.ndarray, xr.DataArray)):
            arr = data.values if isinstance(data, xr.DataArray) else data
            size_mb = arr.nbytes / 1024 / 1024
            log_str += f"  {name:.<40} shape={arr.shape}, dtype={arr.dtype}, size={size_mb:.2f} MB\n"
    print(log_str)

def log_memory_checkpoint(checkpoint_name, verbose=True):
    """Log memory usage at a checkpoint."""
    mem_mb = get_memory_usage()
    if mem_mb is not None and verbose:
        print(f"  [{checkpoint_name:.<35}] Memory: {mem_mb:.2f} MB")
    return mem_mb


# ============================================================================
# HELPER FUNCTIONS (from full_metrics.py)
# ============================================================================

def get_min_max_idx(arr: np.ndarray, axs: int = 1, pad: bool = True) -> np.ndarray:
    """Find local minima and maxima in array."""
    grad = np.diff(arr, axis=axs)
    grad_sign = np.sign(grad)
    min_max = np.abs(np.sign(np.diff(grad_sign, axis=axs)))
    if pad:
        pad_shape = list(min_max.shape)
        pad_shape[axs] = 1
        min_max = np.concatenate([np.zeros(pad_shape), min_max, np.zeros(pad_shape)], axis=axs)
    return min_max


def get_f1_score(min_max_idx_truth: np.ndarray, min_max_idx_ae: np.ndarray, axs: int = 1, kernel_size: int = 10) -> np.ndarray:
    """Compute F1 score for extremum detection."""
    kernel_shape = [1] * min_max_idx_truth.ndim
    kernel_shape[axs] = kernel_size
    kernel = np.ones(kernel_shape)
    truth_expanded = convolve(min_max_idx_truth, kernel, mode='constant', cval=0.0)
    ae_expanded = convolve(min_max_idx_ae, kernel, mode='constant', cval=0.0)
    
    true_positives = (truth_expanded > 0) & (min_max_idx_ae > 0)
    num_true_positives = np.sum(true_positives, axis=axs)
    false_positives = (truth_expanded == 0) & (min_max_idx_ae > 0)
    num_false_positives = np.sum(false_positives, axis=axs)
    false_negatives = (min_max_idx_truth > 0) & (ae_expanded == 0)
    num_false_negatives = np.sum(false_negatives, axis=axs)
    
    precision_den = num_true_positives + num_false_positives
    recall_den = num_true_positives + num_false_negatives
    precision_score = np.where(precision_den == 0, 0, num_true_positives / precision_den)
    recall_score = np.where(recall_den == 0, 0, num_true_positives / recall_den)
    sum_scores = precision_score + recall_score
    f1_score = np.where(sum_scores == 0, 0, 2 * (precision_score * recall_score) / sum_scores)
    
    return f1_score


def get_extremum_position_error(ssp_truth, ssp_ae, depth: np.ndarray, sample: int = 1) -> np.ndarray:
    """Compute error in extremum positions."""
    def cdist_extremum(profile_idx, flat_truth_ext, flat_ae_ext, depth):
        truth_idx = np.where(flat_truth_ext[profile_idx])[0]
        ae_idx = np.where(flat_ae_ext[profile_idx])[0]
        
        if len(truth_idx) == 0 or len(ae_idx) == 0:
            return 0.0
        
        D = cdist(depth[truth_idx][:, None], depth[ae_idx][:, None])
        return D.min(axis=1).mean()
    
    ext_truth = get_min_max_idx(ssp_truth, axs=1, pad=False).astype(bool)
    ext_ae = get_min_max_idx(ssp_ae, axs=1, pad=False).astype(bool)
    
    flat_truth_ext = ext_truth.transpose(0, 2, 3, 1).reshape(-1, ext_truth.shape[1])[::sample]
    flat_ae_ext = ext_ae.transpose(0, 2, 3, 1).reshape(-1, ext_ae.shape[1])[::sample]
    
    error_arr = Parallel(n_jobs=-1)(
        delayed(cdist_extremum)(prof, flat_truth_ext, flat_ae_ext, depth)
        for prof in tqdm(range(flat_truth_ext.shape[0]), mininterval=300.0,desc="Computing extremum position error")
    )

    if sample > 1:
        return np.array(error_arr)
    
    error_arr = np.array(error_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    del flat_truth_ext, flat_ae_ext
    gc.collect()
    return error_arr


def get_dtw_arr(ssp_truth: np.ndarray, ssp_ae: np.ndarray, sample: int = 1) -> np.ndarray:
    """Compute Dynamic Time Warping distances."""
    flatten_truth = ssp_truth.transpose(0, 2, 3, 1).reshape(-1, ssp_truth.shape[1])[::sample]
    flatten_ae = ssp_ae.transpose(0, 2, 3, 1).reshape(-1, ssp_ae.shape[1])[::sample]
    
    dtw_arr = Parallel(n_jobs=-1)(
        delayed(dtw.distance)(flatten_ae[prof], flatten_truth[prof])
        for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing DTW")
    )

    # for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing DTW"):
    #     dtw_val = dtw.distance(flatten_ae[prof], flatten_truth[prof])
    #     if prof == 0:
    #         dtw_arr = np.array([dtw_val])
    #     else:
    #         dtw_arr = np.append(dtw_arr, dtw_val)
    
    if sample > 1:
        return np.array(dtw_arr)
    
    dtw_arr = np.array(dtw_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    del flatten_truth, flatten_ae
    gc.collect()
    return dtw_arr


def get_wd_freq_arr(power_truth: np.ndarray, power_ae: np.ndarray, freqs: np.ndarray, sample: int = 1) -> np.ndarray:
    """Compute Wasserstein distances."""
    flatten_truth = power_truth.transpose(0, 2, 3, 1).reshape(-1, power_truth.shape[1])[::sample]
    flatten_ae = power_ae.transpose(0, 2, 3, 1).reshape(-1, power_ae.shape[1])[::sample]
    
    wd_arr = Parallel(n_jobs=-1)(
        delayed(wasserstein_distance)(freqs, freqs, flatten_ae[prof], flatten_truth[prof])
        for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing Wasserstein distance")
    )

    # for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing Wasserstein distance"):
    #     wd_val = wasserstein_distance(freqs, freqs, flatten_ae[prof], flatten_truth[prof])
    #     if prof == 0:
    #         wd_arr = np.array([wd_val])
    #     else:
    #         wd_arr = np.append(wd_arr, wd_val)

    if sample > 1:
        return np.array(wd_arr)
    
    wd_arr = np.array(wd_arr).reshape(power_truth.shape[0], power_truth.shape[2], power_truth.shape[3])
    del flatten_truth, flatten_ae
    gc.collect()
    return wd_arr


def get_wd_arr(ssp_truth: np.ndarray, ssp_ae: np.ndarray, sample: int = 1) -> np.ndarray:
    """Compute Wasserstein distances on SSP profiles along the depth axis."""
    flatten_truth = ssp_truth.transpose(0, 2, 3, 1).reshape(-1, ssp_truth.shape[1])[::sample]
    flatten_ae = ssp_ae.transpose(0, 2, 3, 1).reshape(-1, ssp_ae.shape[1])[::sample]
    depth = np.arange(ssp_truth.shape[1], dtype=np.float64)

    def _to_valid_weights(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x - np.min(x)
        weight_sum = x.sum()
        if weight_sum <= 0:
            return np.ones_like(x, dtype=np.float64) / x.size
        return x / weight_sum

    wd_arr = Parallel(n_jobs=-1)(
        delayed(wasserstein_distance)(
            depth,
            depth,
            _to_valid_weights(flatten_ae[prof]),
            _to_valid_weights(flatten_truth[prof]),
        )
        for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing depth Wasserstein distance")
    )

    if sample > 1:
        return np.array(wd_arr)

    wd_arr = np.array(wd_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    del flatten_truth, flatten_ae
    gc.collect()
    return wd_arr


def get_seasonal_time_indices(da_time, season: str):
    """
    Get time indices for a specific season.
    
    Parameters
    ----------
    da_time : xr.DataArray or np.ndarray or list
        Time coordinate from xarray DataArray
    season : str
        Season name: 'all', 'spring', 'summer', 'autumn', 'winter'
        
    Returns
    -------
    np.ndarray
        Boolean or integer indices for the specified season
    """
    # Convert to pandas datetime if needed
    if hasattr(da_time, 'values'):
        time_vals = pd.to_datetime(da_time.values)
    else:
        time_vals = pd.to_datetime(da_time)
    
    months = time_vals.month
    
    if season == 'all':
        return np.arange(len(time_vals))
    elif season == 'spring':
        # March (3) to May (5)
        return np.where((months >= 3) & (months <= 5))[0]
    elif season == 'summer':
        # June (6) to August (8)
        return np.where((months >= 6) & (months <= 8))[0]
    elif season == 'autumn':
        # September (9) to November (11)
        return np.where((months >= 9) & (months <= 11))[0]
    elif season == 'winter':
        # December (12) and January (1), February (2)
        return np.where((months == 12) | (months == 1) | (months == 2))[0]
    else:
        raise ValueError(f"Unknown season: {season}. Must be one of: 'all', 'spring', 'summer', 'autumn', 'winter'")


def compute_power_spectrum(da, dim="z", detrend=True, window=True):
    """Compute power spectrum along a given dimension."""
    axis = da.get_axis_num(dim)
    n = da.sizes[dim]
    
    z_uniform = np.linspace(float(da.z.min()), float(da.z.max()), n)
    da = da.interp(z=z_uniform)
    
    coords = da[dim].values
    d = float(coords[1] - coords[0])
    data = da.values
    
    if detrend:
        from scipy.signal import detrend as detrend_signal
        data = detrend_signal(data, axis=axis)
    
    window_correction = 1.0
    if window:
        from scipy.signal.windows import hann
        win = hann(n)
        shape = [1] * data.ndim
        shape[axis] = n
        data = data * win.reshape(shape)
        window_correction = n / np.sum(win**2)
    else:
        window_correction = 1.0
    
    fft_vals = np.fft.rfft(data, axis=axis)
    power = (np.abs(fft_vals) ** 2) / (n * d) * window_correction
    freqs = np.fft.rfftfreq(n, d=d)
    
    dims = list(da.dims)
    dims[axis] = "freq"
    
    coords = dict(da.coords)
    coords["freq"] = freqs
    coords.pop(dim)
    
    power_da = xr.DataArray(power, dims=dims, coords=coords, name="power_spectrum")
    return power_da, freqs


def ssim_1d(x, y, sigma=1.5, C1=1e-4, C2=1e-4):
    """Compute 1D SSIM for profiles."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    mu_x = gaussian_filter1d(x, sigma)
    mu_y = gaussian_filter1d(y, sigma)
    
    sigma_x = gaussian_filter1d(x**2, sigma) - mu_x**2
    sigma_y = gaussian_filter1d(y**2, sigma) - mu_y**2
    sigma_xy = gaussian_filter1d(x*y, sigma) - mu_x*mu_y
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    
    return np.mean(numerator / (denominator + 1e-12))


def ms_ssim_1d(x, y, scales=5):
    """Compute Multi-scale SSIM for profiles."""
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weights = weights[:scales]
    
    mssim = []
    x_s, y_s = x, y
    
    for i in range(scales):
        mssim.append(ssim_1d(x_s, y_s))
        if i < scales - 1:
            x_s = x_s[::2]
            y_s = y_s[::2]
    
    mssim = np.array(mssim)
    return np.prod(mssim ** weights)


def compute_ssim(ssp_truth: np.ndarray, ssp_ae: np.ndarray, sample: int = 1) -> np.ndarray:
    """Compute SSIM for all profiles."""
    flatten_truth = ssp_truth.transpose(0, 2, 3, 1).reshape(-1, ssp_truth.shape[1])[::sample]
    flatten_ae = ssp_ae.transpose(0, 2, 3, 1).reshape(-1, ssp_ae.shape[1])[::sample]

    ssim_arr = Parallel(n_jobs=-1)(
        delayed(ssim_1d)(flatten_truth[prof], flatten_ae[prof])
        for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing SSIM")
    )

    # for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing SSIM"):
    #     ssim_val = ssim_1d(flatten_truth[prof], flatten_ae[prof])
    #     if prof == 0:
    #         ssim_arr = np.array([ssim_val])
    #     else:
    #         ssim_arr = np.append(ssim_arr, ssim_val)

    if sample > 1:
        return np.array(ssim_arr)
    
    ssim_arr = np.array(ssim_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    del flatten_truth, flatten_ae
    gc.collect()
    return ssim_arr


def compute_ms_ssim(ssp_truth: np.ndarray, ssp_ae: np.ndarray, sample: int = 1) -> np.ndarray:
    """Compute MS-SSIM for all profiles."""
    flatten_truth = ssp_truth.transpose(0, 2, 3, 1).reshape(-1, ssp_truth.shape[1])[::sample]
    flatten_ae = ssp_ae.transpose(0, 2, 3, 1).reshape(-1, ssp_ae.shape[1])[::sample]

    mssim_arr = Parallel(n_jobs=-1)(
        delayed(ms_ssim_1d)(flatten_truth[prof], flatten_ae[prof])
        for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing MS-SSIM")
    )
    
    # for prof in tqdm(range(flatten_truth.shape[0]), mininterval=300.0, desc="Computing MS-SSIM"):
    #     mssim_val = ms_ssim_1d(flatten_truth[prof], flatten_ae[prof])
    #     if prof == 0:
    #         mssim_arr = np.array([mssim_val])
    #     else:
    #         mssim_arr = np.append(mssim_arr, mssim_val)

    if sample > 1:
        return np.array(mssim_arr)

    mssim_arr = np.array(mssim_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    del flatten_truth, flatten_ae
    gc.collect()
    return mssim_arr




def apply_pooling_upsample_to_pca(pca_ae: np.ndarray, ssp_truth_shape: tuple, n_layer_pooling: int, pooling_mode: str = "mean") -> np.ndarray:
    """
    Apply spatial pooling and upsampling to PCA-transformed data using numpy/scipy, then inverse transform.
    Works entirely with numpy arrays using skimage.measure.block_reduce and scipy.interpolate.RectBivariateSpline.
    
    Args:
        pca_ae: flattened PCA-transformed array with shape (batch*lat*lon, n_components)
        ssp_truth_shape: tuple of original spatial shape (batch, depth, lat, lon)
        n_layer_pooling: number of pooling layers to apply (e.g., 3)
        pca: fitted PCA object with inverse_transform method
        pooling_mode: "mean" for average pooling or "max" for max pooling
        
    Returns:
        reconstructed: data in shape (batch, depth, lat, lon) after pooling/upsampling and inverse transform
    """
    from skimage.measure import block_reduce
    from scipy.interpolate import RectBivariateSpline
    
    batch, depth, lat, lon = ssp_truth_shape
    n_components = pca_ae.shape[1]
    
    # Reshape from (batch*lat*lon, n_components) to (batch, lat, lon, n_components)
    pca_ae_reshaped = pca_ae.reshape(batch, lat, lon, n_components)
    
    # Transpose to (batch, n_components, lat, lon) for spatial pooling
    pca_ae_spatial = pca_ae_reshaped.transpose(0, 3, 1, 2)  # (batch, n_components, lat, lon)
    
    # Apply pooling and upsampling for each batch and component
    pooled_upsampled_list = []
    
    for b in range(batch):
        component_list = []
        for c in range(n_components):
            data_2d = pca_ae_spatial[b, c, :, :]  # (lat, lon)
            
            # Apply pooling layers
            pooled_data = data_2d.copy()
            for _ in range(n_layer_pooling):
                block_size = 2
                if pooling_mode == "mean":
                    pooled_data = block_reduce(pooled_data, block_size=(block_size, block_size), func=np.mean)
                else:  # max
                    pooled_data = block_reduce(pooled_data, block_size=(block_size, block_size), func=np.max)
            
            # Upsample back to original size using RectBivariateSpline
            pooled_h, pooled_w = pooled_data.shape

            
            # Create coordinate arrays for the pooled data
            y_old = np.arange(pooled_h)
            x_old = np.arange(pooled_w)
            
            # Create RectBivariateSpline interpolator (cubic by default, kx=ky=3)
            spl = RectBivariateSpline(y_old, x_old, pooled_data, kx=min(3, pooled_data.shape[0]-1), ky=min(3, pooled_data.shape[1]-1))
            
            # Create new coordinate arrays for upsampled data
            y_new = np.linspace(0, pooled_h - 1, lat)
            x_new = np.linspace(0, pooled_w - 1, lon)
            
            # Interpolate to original size
            upsampled_data = spl(y_new, x_new, grid=True)
            
            component_list.append(upsampled_data)
        
        # Stack components and convert to (n_components, lat, lon)
        pooled_upsampled_list.append(np.array(component_list))
    
    # Concatenate batches: (batch, n_components, lat, lon)
    pooled_upsampled_np = np.array(pooled_upsampled_list)
    
    # Reshape to flattened form (batch*lat*lon, n_components) for inverse transform
    pooled_upsampled_flat = pooled_upsampled_np.transpose(0, 2, 3, 1).reshape(-1, n_components)
    
    return pooled_upsampled_flat


# ============================================================================
# BEST AND WORST SAMPLES FUNCTIONS
# ============================================================================

def get_best_worst_random(metric_arr, truth_arr, ae_arr, metric_name,best_is_min=True):

    flat_score = metric_arr.flatten()

    if best_is_min:
        best_idx = np.argmin(metric_arr)
        worst_idx = np.argmax(metric_arr)
    else:
        best_idx = np.argmax(metric_arr)
        worst_idx = np.argmin(metric_arr)

    
    def unravel(idx):
        return np.unravel_index(idx, metric_arr.shape)

    def extract(idx):
        t0, lat0, lon0 = idx
        return {
            metric_name: np.abs(metric_arr[t0, lat0, lon0]),
            #'t_lat': ((t0, lat0), truth_arr[t0, :, lat0, :], ae_arr[t0, :, lat0, :].copy()),  # .copy() to avoid keeping ref to full array
            't_lat_lon': ((t0, lat0, lon0), truth_arr[t0, :, lat0, lon0], ae_arr[t0, :, lat0, lon0].copy())
        }

    return {'best': extract(unravel(best_idx)), 'worst': extract(unravel(worst_idx))}


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_metrics_batched(
    ssp_truth_da: xr.DataArray,
    ssp_ae_da: xr.DataArray,
    depth_array: np.ndarray,
    data_dict_metrics: List[str],
    pca: Optional[PCA] = None,
    batch_size: int = 5,
    verbose: bool = False,
) -> Dict:
    """
    Compute metrics on large SSP data in batches to avoid OOM issues.
    
    Parameters
    ----------
    ssp_truth_da : xr.DataArray
        True SSP data array with shape (time, z, lat, lon)
    ssp_ae_da : xr.DataArray
        Reconstructed SSP data array with same shape
    depth_array : np.ndarray
        Depth array in meters
    data_dict_metrics : List[str]
        List of metrics to compute
    pca : Optional[PCA]
        Pre-fitted PCA object
    batch_size : int
        Number of time steps to process per batch (default: 5)
    verbose : bool
        If True, print progress information
        
    Returns
    -------
    Dict
        Dictionary containing computed metrics and their statistics
    """
    n_time = ssp_truth_da.shape[0]
    metrics_dict = {}
    data = {"selected": {}}
    
    if verbose:
        print(f"\nComputing metrics for shape: {ssp_truth_da.shape} in batches of {batch_size}")
    
    # Accumulators for batch-wise statistics
    accumulators = {
        'rmse_sum': 0.0,
        'rmse_count': 0,
        'mae_sum': 0.0,
        'mae_count': 0,
        'mse_sum': 0.0,
        'mse_count': 0,
        'psnr_sum': 0.0,
        'psnr_count': 0,
        'ecs_arr': [],
        'extremum_pos_arr': [],
        'f1_score_arr': [],
        'pearson_corr_arr': [],
        'r2_res': 0.0,
        'r2_tot': 0.0,
        'dtw_arr': [],
        'lsd_arr': [],
        'peak_freq_error_arr': [],
        'wasserstein_arr': [],
        'ssim_arr': [],
        'mssim_arr': [],
        'pae_pca': None,
    }
    
    # First pass: compute global statistics and per-profile metrics
    for batch_idx in tqdm(range(0, n_time, batch_size), desc="Computing metrics", disable=not verbose):
        end_idx = min(batch_idx + batch_size, n_time)
        
        # Extract batch
        ssp_truth_batch = ssp_truth_da.isel(time=slice(batch_idx, end_idx))
        ssp_ae_batch = ssp_ae_da.isel(time=slice(batch_idx, end_idx))
        
        ssp_truth_np = ssp_truth_batch.values.astype(np.float32)
        ssp_ae_np = ssp_ae_batch.values.astype(np.float32)
        
        batch_size_actual = ssp_truth_np.shape[0]
        
        # === RMSE ===
        if verbose:
            print(f"  Batch {batch_idx//batch_size + 1}: Computing RMSE...")
        batch_rmse = np.sqrt(((ssp_ae_np - ssp_truth_np) ** 2).mean())
        batch_mse = ((ssp_ae_np - ssp_truth_np) ** 2).mean()
        accumulators['rmse_sum'] += batch_rmse * batch_size_actual
        accumulators['rmse_count'] += batch_size_actual
        accumulators['mse_sum'] += batch_mse * batch_size_actual
        accumulators['mse_count'] += batch_size_actual
        
        # === MAE ===
        batch_mae = np.abs(ssp_ae_np - ssp_truth_np).mean(axis=1, keepdims=False).mean()
        accumulators['mae_sum'] += batch_mae * batch_size_actual
        accumulators['mae_count'] += batch_size_actual
        
        # === PSNR ===
        max_val_ssp = float(np.nanmax(ssp_truth_da.values))
        batch_mse_per_loc = ((ssp_ae_np - ssp_truth_np) ** 2).mean(axis=1)
        batch_psnr = 10 * np.log10((max_val_ssp ** 2) / (batch_mse_per_loc.mean() + 1e-10))
        accumulators['psnr_sum'] += batch_psnr * batch_size_actual
        accumulators['psnr_count'] += batch_size_actual
        
        # === ECS ===
        max_ssp_truth_idx = np.nanargmax(ssp_truth_np, axis=1)
        max_ssp_ae_idx = np.nanargmax(ssp_ae_np, axis=1)
        batch_ecs = np.abs(depth_array[max_ssp_truth_idx] - depth_array[max_ssp_ae_idx])
        accumulators['ecs_arr'].append(batch_ecs)
        
        # === Extremum Position Error ===
        batch_extremum_pos = get_extremum_position_error(ssp_truth_np, ssp_ae_np, depth_array)
        accumulators['extremum_pos_arr'].append(batch_extremum_pos)
        
        # === F1 Score ===
        min_max_idx_truth = get_min_max_idx(ssp_truth_np, axs=1, pad=False)
        min_max_idx_ae = get_min_max_idx(ssp_ae_np, axs=1, pad=False)
        batch_f1 = get_f1_score(min_max_idx_truth, min_max_idx_ae, axs=1, kernel_size=10)
        accumulators['f1_score_arr'].append(batch_f1)
        
        # === Pearson Correlation ===
        pears = pearsonr(
            ssp_truth_np.transpose(1, 0, 2, 3).reshape(ssp_truth_np.shape[1], -1),
            ssp_ae_np.transpose(1, 0, 2, 3).reshape(ssp_ae_np.shape[1], -1)
        )
        batch_pearson = pears.statistic.reshape(batch_size_actual, ssp_truth_np.shape[2], ssp_truth_np.shape[3])
        accumulators['pearson_corr_arr'].append(batch_pearson)
        
        # === R² Score ===
        batch_ss_res = ((ssp_ae_np - ssp_truth_np) ** 2).sum()
        batch_ss_tot = ((ssp_truth_np - ssp_truth_np.mean()) ** 2).sum()
        accumulators['r2_res'] += batch_ss_res
        accumulators['r2_tot'] += batch_ss_tot
        
        # === DTW ===
        batch_dtw = get_dtw_arr(ssp_truth_np, ssp_ae_np, sample=100)
        accumulators['dtw_arr'].append(batch_dtw)
        
        # === Power Spectrum Metrics ===
        if verbose:
            print(f"  Batch {batch_idx//batch_size + 1}: Computing power spectrum metrics...")
        power_truth, freqs = compute_power_spectrum(ssp_truth_batch, dim="z", detrend=True, window=True)
        power_ae, _ = compute_power_spectrum(ssp_ae_batch, dim="z", detrend=True, window=True)
        
        eps = 1e-12
        log_truth = np.log(power_truth + eps)
        log_ae = np.log(power_ae + eps)
        batch_lsd = np.sqrt(np.mean((log_truth - log_ae)**2, axis=1))
        accumulators['lsd_arr'].append(batch_lsd)
        
        peak_truth = np.argmax(power_truth, axis=1)
        peak_ae = np.argmax(power_ae, axis=1)
        batch_peak_freq = np.abs(freqs[peak_truth] - freqs[peak_ae])
        accumulators['peak_freq_error_arr'].append(batch_peak_freq)
        
        batch_wd = get_wd_freq_arr(power_truth, power_ae, freqs, sample=1)
        accumulators['power_wasserstein_arr'].append(batch_wd)
        
        del power_truth, power_ae
        gc.collect()
        
        # === SSIM ===
        if verbose:
            print(f"  Batch {batch_idx//batch_size + 1}: Computing SSIM...")
        batch_ssim = compute_ssim(ssp_truth_np, ssp_ae_np, sample=1)
        accumulators['ssim_arr'].append(batch_ssim)
        
        # === MS-SSIM ===
        batch_mssim = compute_ms_ssim(ssp_truth_np, ssp_ae_np, sample=100)
        accumulators['mssim_arr'].append(batch_mssim)
        
        # === PCA Transform (if needed) ===
        if pca is not None:
            batch_ae_pca = pca.transform(ssp_ae_np.transpose(0, 2, 3, 1).reshape(-1, ssp_ae_np.shape[1]))
            if accumulators['pae_pca'] is None:
                accumulators['pae_pca'] = batch_ae_pca
            else:
                accumulators['pae_pca'] = np.vstack([accumulators['pae_pca'], batch_ae_pca])
    
    # === Aggregate results ===
    if verbose:
        print("\nAggregating batch results...")
    
    metrics_dict["rmse"] = float(accumulators['rmse_sum'] / accumulators['rmse_count']) if accumulators['rmse_count'] > 0 else 0.0
    metrics_dict["mae"] = float(accumulators['mae_sum'] / accumulators['mae_count']) if accumulators['mae_count'] > 0 else 0.0
    metrics_dict["psnr"] = float(accumulators['psnr_sum'] / accumulators['psnr_count']) if accumulators['psnr_count'] > 0 else 0.0
    
    # Concatenate per-profile metrics
    ecs_arr = np.concatenate(accumulators['ecs_arr']) if accumulators['ecs_arr'] else np.array([])
    extremum_pos_arr = np.concatenate(accumulators['extremum_pos_arr']) if accumulators['extremum_pos_arr'] else np.array([])
    f1_score_arr = np.concatenate(accumulators['f1_score_arr']) if accumulators['f1_score_arr'] else np.array([])
    pearson_arr = np.concatenate(accumulators['pearson_corr_arr']) if accumulators['pearson_corr_arr'] else np.array([])
    dtw_arr = np.concatenate(accumulators['dtw_arr']) if accumulators['dtw_arr'] else np.array([])
    lsd_arr = np.concatenate(accumulators['lsd_arr']) if accumulators['lsd_arr'] else np.array([])
    peak_freq_arr = np.concatenate(accumulators['peak_freq_error_arr']) if accumulators['peak_freq_error_arr'] else np.array([])
    wasserstein_arr = np.concatenate(accumulators['wasserstein_arr']) if accumulators['wasserstein_arr'] else np.array([])
    ssim_arr = np.concatenate(accumulators['ssim_arr']) if accumulators['ssim_arr'] else np.array([])
    mssim_arr = np.concatenate(accumulators['mssim_arr']) if accumulators['mssim_arr'] else np.array([])
    
    metrics_dict["ecs"] = float(np.mean(ecs_arr)) if len(ecs_arr) > 0 else 0.0
    metrics_dict["extremum_pos"] = float(np.mean(extremum_pos_arr)) if len(extremum_pos_arr) > 0 else 0.0
    metrics_dict["f1_score"] = float(np.mean(f1_score_arr)) if len(f1_score_arr) > 0 else 0.0
    metrics_dict["pearson"] = float(np.mean(pearson_arr)) if len(pearson_arr) > 0 else 0.0
    metrics_dict["pearson_pvalue"] = 0.0  # Placeholder - would need per-batch p-values
    
    # R² Score aggregation
    r2_score = 1 - (accumulators['r2_res'] / accumulators['r2_tot']) if accumulators['r2_tot'] != 0 else 0.0
    metrics_dict["r2_score"] = float(r2_score)
    
    metrics_dict["dtw"] = float(np.mean(dtw_arr)) if len(dtw_arr) > 0 else 0.0
    metrics_dict["lsd"] = float(np.mean(lsd_arr)) if len(lsd_arr) > 0 else 0.0
    metrics_dict["peak_freq_error"] = float(np.mean(peak_freq_arr)) if len(peak_freq_arr) > 0 else 0.0
    metrics_dict["wasserstein"] = float(np.mean(wasserstein_arr)) if len(wasserstein_arr) > 0 else 0.0
    metrics_dict["ssim"] = float(np.mean(ssim_arr)) if len(ssim_arr) > 0 else 0.0
    metrics_dict["ms_ssim"] = float(np.mean(mssim_arr)) if len(mssim_arr) > 0 else 0.0
    
    # === NSR (must process full dataset) ===
    if verbose:
        print("\n" + "="*70)
        print("Computing NSR (spectral resolution)...")
        print("="*70)
        log_memory_checkpoint("Before NSR computation", verbose=True)
        log_data_shapes({
            'ssp_truth_da': ssp_truth_da,
            'ssp_ae_da': ssp_ae_da
        }, "NSR Input Data Shapes")
    
    try:
        for idx_t, t_r in enumerate([0.1, 0.5, 0.9], 1):
            if verbose:
                print(f"\n  [{idx_t}/3] Computing NSR with target_ratio={t_r}...")
                log_memory_checkpoint(f"Before nsr_depth (t_r={t_r})", verbose=True)
            
            nsr_depth = compute_nsr_along_depth(ssp_truth_da, ssp_ae_da, target_ratio=t_r)
            
            if verbose:
                log_memory_checkpoint(f"After nsr_depth (t_r={t_r})", verbose=True)
                print(f"    nsr_depth result: {nsr_depth}")
            
            nsr_map = compute_nsr_spatial_map(ssp_truth_da, ssp_ae_da, target_ratio=t_r)
            
            if verbose:
                log_memory_checkpoint(f"After nsr_map (t_r={t_r})", verbose=True)
                print(f"    nsr_map result: {nsr_map}")
            
            metrics_dict[f"nsr_depth_resolution_{int(t_r*100)}"] = float(nsr_depth["resolution"])
            metrics_dict[f"nsr_spatial_resolution_{int(t_r*100)}"] = float(nsr_map["resolution"])
            
            if verbose:
                print(f"    ✓ Completed NSR for t_r={t_r}")
    except Exception as e:
        if verbose:
            print(f"\n  ✗ Error during NSR computation: {e}")
            log_memory_checkpoint("After NSR error", verbose=True)
            import traceback
            traceback.print_exc()
        raise
    
    if verbose:
        log_memory_checkpoint("After all NSR computations", verbose=True)
        gc.collect()
        log_memory_checkpoint("After garbage collection", verbose=True)
    
    # === PCA Reconstruction ===
    if pca is not None and accumulators['pae_pca'] is not None:
        ssp_truth_np = ssp_truth_da.values.astype(np.float32)
        ssp_ae_np = ssp_ae_da.values.astype(np.float32)
        
        for n_components in [3, 6]:
            ae_pca_n = accumulators['pae_pca'][:, :n_components]
            ssp_ae_pca_recon = ae_pca_n @ pca.components_[:n_components, :] + pca.mean_
            ssp_ae_pca_recon = ssp_ae_pca_recon.reshape(
                ssp_ae_np.shape[0], ssp_ae_np.shape[2], ssp_ae_np.shape[3], ssp_ae_np.shape[1]
            ).transpose(0, 3, 1, 2)
            rmse_pca = float(np.sqrt(((ssp_ae_pca_recon - ssp_truth_np) ** 2).mean()))
            metrics_dict[f"rmse_pca_{n_components}_components"] = rmse_pca
    
    if verbose:
        print(f"\n✓ Computed {len(metrics_dict)} metric values")
    
    return metrics_dict, data


def compute_metrics(
    ssp_truth_da: xr.DataArray,
    ssp_ae_da: xr.DataArray,
    depth_array: np.ndarray,
    data_dict_metrics: List[str],
    pca: Optional[PCA] = None,
    verbose: bool = False,
) -> Dict:
    """
    Compute specified metrics between truth and reconstructed SSP data.
    
    Parameters
    ----------
    ssp_truth_da : xr.DataArray
        True SSP data array with shape (time, z, lat, lon)
    ssp_ae_da : xr.DataArray
        Reconstructed SSP data array with same shape
    depth_array : np.ndarray
        Depth array in meters
    data_dict_metrics : List[str]
        List of metrics to compute. Options:
        ["RMSE", "MAE", "PSNR", "ECS", "EXTREMUM_POS_ERROR", "F1_SCORE",
         "PEARSON", "R2_SCORE", "DTW", "LSD", "PEAK_FREQ_ERROR", 
         "WASSERSTEIN", "SSIM", "MS_SSIM", "NSR"]
    compute_pca : bool, optional
        If True, also compute PCA-based metrics
    verbose : bool, optional
        If True, print progress information
        
    Returns
    -------
    dict
        Dictionary containing computed metrics and their statistics
    """
    metrics_dict = {}

    
    # Convert to numpy arrays
    ssp_truth = ssp_truth_da.values.astype(np.float32)
    ssp_ae = ssp_ae_da.values.astype(np.float32)

    data = {"selected":{}}
    # Note: T0, LAT0, LON0 sampling removed to avoid undefined variables
    # Can be re-added if specific indices are needed
    
    if verbose:
        print(f"\nComputing metrics for shape: {ssp_truth.shape}")
        #print(f"Available metrics: {data_dict_metrics}\n")
    
    # RMSE
    if verbose:
        print("  Computing RMSE...")
    metrics_dict["rmse"] = float(np.sqrt(((ssp_ae_da - ssp_truth_da) ** 2).mean()))
    if "RMSE" in data_dict_metrics:
        metric_arr = np.sqrt(((ssp_ae_da - ssp_truth_da) ** 2).mean(dim="z", skipna=True)).data 
        data["RMSE"] = get_best_worst_random(metric_arr, ssp_truth, ssp_ae, "RMSE", best_is_min=True)

    # MAE
    if verbose:
        print("  Computing MAE...")
    mae_da = np.abs(ssp_ae_da - ssp_truth_da).mean(dim="z", skipna=True)
    metrics_dict["mae"] = float(mae_da.mean())
    if "MAE" in data_dict_metrics:
        data["MAE"] = get_best_worst_random(mae_da.values, ssp_truth, ssp_ae, "MAE", best_is_min=True)

    
    # PSNR
    if verbose:
        print("  Computing PSNR...")
    mse_per_location = ((ssp_ae_da - ssp_truth_da) ** 2).mean(dim="z", skipna=True)
    global_mse = float(np.nanmean(mse_per_location))
    max_val_ssp = float(np.nanmax(ssp_truth_da.values))
    global_psnr = 10 * np.log10((max_val_ssp ** 2) / (global_mse + 1e-10))
    metrics_dict["psnr"] = float(global_psnr)

    if "PSNR" in data_dict_metrics:
        max_val_ssp = float(np.nanmax(ssp_truth_da.values))
        psnr_arr = 10 * np.log10(max_val_ssp**2 / mse_per_location + 1e-10).data
        # psnr_da = psnr_da.where(np.isfinite(psnr_da), np.nan)
        # metrics_dict["psnr"] = float(np.nanmean(psnr_da))
        data["PSNR"] = get_best_worst_random(psnr_arr, ssp_truth, ssp_ae, "PSNR", best_is_min=False)
    

    # ECS (Extremum Closest to Surface)
    if verbose:
        print("  Computing ECS...")
    max_ssp_truth_idx = np.nanargmax(ssp_truth, axis=1)
    max_ssp_ae_idx = np.nanargmax(ssp_ae, axis=1)
    ecs = np.abs(depth_array[max_ssp_truth_idx] - depth_array[max_ssp_ae_idx])
    metrics_dict["ecs"] = float(np.mean(ecs))
    if "ECS" in data_dict_metrics:
        data["ECS"] = get_best_worst_random(ecs, ssp_truth, ssp_ae, "ECS", best_is_min=True)

    # Extremum Position Error
    if verbose:
        print("  Computing Extremum Position Error...")
    extremum_position_error_arr = get_extremum_position_error(ssp_truth, ssp_ae, depth_array, sample=10)
    metrics_dict["extremum_pos"] = float(np.mean(extremum_position_error_arr))
    if "EXTREMUM_POS_ERROR" in data_dict_metrics and extremum_position_error_arr.shape[0] == ssp_truth.shape[0]:  # Only compute best/worst if we have per-profile values
        data["EXTREMUM_POS_ERROR"] = get_best_worst_random(extremum_position_error_arr, ssp_truth, ssp_ae, "EXTREMUM_POS_ERROR", best_is_min=True)

    
    # F1 Score
    if verbose:
        print("  Computing F1 Score...")
    min_max_idx_truth = get_min_max_idx(ssp_truth, axs=1, pad=False)
    min_max_idx_ae = get_min_max_idx(ssp_ae, axs=1, pad=False)
    f1_score = get_f1_score(min_max_idx_truth, min_max_idx_ae, axs=1, kernel_size=10)
    metrics_dict["f1_score"] = float(np.mean(f1_score))
    if "F1_SCORE" in data_dict_metrics:
        data["F1_SCORE"] = get_best_worst_random(f1_score, ssp_truth, ssp_ae, "F1_SCORE", best_is_min=False)

    # Pearson Correlation
    if verbose:
        print("  Computing Pearson Correlation...")
    pears = pearsonr(
        ssp_truth.transpose(1, 0, 2, 3).reshape(ssp_truth.shape[1], -1),
        ssp_ae.transpose(1, 0, 2, 3).reshape(ssp_ae.shape[1], -1)
    )
    pearson_corr = pears.statistic.reshape(ssp_truth.shape[0],ssp_truth.shape[2], ssp_truth.shape[3])
    metrics_dict["pearson"] = float(pearson_corr.mean())
    metrics_dict["pearson_pvalue"] = float(pears.pvalue.mean())
    if "PEARSON" in data_dict_metrics:
        data["PEARSON"] = get_best_worst_random(pearson_corr, ssp_truth, ssp_ae, "PEARSON", best_is_min=False)
    
    # R² Score
    if verbose:
        print("  Computing R² Score...")
    ss_res = ((ssp_ae_da - ssp_truth_da) ** 2).sum()
    ss_tot = ((ssp_truth_da - ssp_truth_da.mean()) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot)
    metrics_dict["r2_score"] = float(r2_score.values)
    if "R2_SCORE" in data_dict_metrics:
        r2_score_arr = 1 - (((ssp_ae_da - ssp_truth_da) ** 2).sum(dim="z", skipna=True) / ((ssp_truth_da - ssp_truth_da.mean(dim="z", skipna=True)) ** 2).sum(dim="z", skipna=True))
        data["R2_SCORE"] = get_best_worst_random(r2_score_arr.values, ssp_truth, ssp_ae, "R2_SCORE", best_is_min=False)
    
    # DTW

    if verbose:
        print("  Computing DTW...")
    dtw_arr = get_dtw_arr(ssp_truth, ssp_ae, sample=10)
    metrics_dict["dtw"] = float(np.mean(dtw_arr))
    if "DTW" in data_dict_metrics and dtw_arr.shape[0] == ssp_truth.shape[0]:  # Only compute best/worst if we have per-profile DTW values
        data["DTW"] = get_best_worst_random(dtw_arr, ssp_truth, ssp_ae, "DTW", best_is_min=True)
    del dtw_arr
    gc.collect()


    # warseinstein

    wd_arr = get_wd_arr(ssp_truth, ssp_ae, sample=1)
    metrics_dict["wasserstein"] = float(np.mean(wd_arr))
    if "WASSERSTEIN" in data_dict_metrics and wd_arr.shape[0] == ssp_truth.shape[0]:  # Only compute best/worst if we have per-profile Wasserstein values
        data["WASSERSTEIN"] = get_best_worst_random(wd_arr, ssp_truth, ssp_ae, "WASSERSTEIN", best_is_min=True)
    del wd_arr
    gc.collect()



    # Power spectrum metrics (LSD, Peak Freq Error, Wasserstein)

    if verbose:
        print("  Computing power spectrum metrics...")
    power_da_truth, freqs = compute_power_spectrum(ssp_truth_da, dim="z", detrend=True, window=True)
    power_da_ae, _ = compute_power_spectrum(ssp_ae_da, dim="z", detrend=True, window=True)
    
    
    eps = 1e-12
    log_truth = np.log(power_da_truth + eps)
    log_rec = np.log(power_da_ae + eps)
    lsd = np.sqrt(np.mean((log_truth - log_rec)**2))
    metrics_dict["lsd"] = float(lsd)
    if "LSD" in data_dict_metrics:
        lsd_arr = np.sqrt(np.mean((log_truth - log_rec)**2, axis=1)).data
        data["LSD"] = get_best_worst_random(lsd_arr, ssp_truth, ssp_ae, "LSD", best_is_min=True)


    peak_truth = np.argmax(power_da_truth.data, axis=1)
    peak_rec = np.argmax(power_da_ae.data, axis=1)
    peak_freq_error = np.abs(freqs[peak_truth] - freqs[peak_rec])
    metrics_dict["peak_freq_error"] = float(np.mean(peak_freq_error))
    if "PEAK_FREQ_ERROR" in data_dict_metrics:
        data["PEAK_FREQ_ERROR"] = get_best_worst_random(peak_freq_error, ssp_truth, ssp_ae, "PEAK_FREQ_ERROR", best_is_min=True)
    
    
    wd_arr = get_wd_freq_arr(power_da_truth.data, power_da_ae.data, freqs, sample=1)
    metrics_dict["power_wasserstein"] = float(np.mean(wd_arr))
    if "POWER_WASSERSTEIN" in data_dict_metrics and wd_arr.shape[0] == ssp_truth.shape[0]:  # Only compute best/worst if we have per-profile Wasserstein values
        data["POWER_WASSERSTEIN"] = get_best_worst_random(wd_arr, ssp_truth, ssp_ae, "POWER_WASSERSTEIN", best_is_min=True)
    del wd_arr, power_da_truth, power_da_ae
    gc.collect()

    # SSIM
    if verbose:
        print("  Computing SSIM...")
    ssim_arr = compute_ssim(ssp_truth, ssp_ae, sample=10)
    metrics_dict["ssim"] = float(np.mean(ssim_arr))
    if "SSIM" in data_dict_metrics and ssim_arr.shape[0] == ssp_truth.shape[0]:  # Only compute best/worst if we have per-profile SSIM values
        data["SSIM"] = get_best_worst_random(ssim_arr, ssp_truth, ssp_ae, "SSIM", best_is_min=False)
    del ssim_arr
    gc.collect()

    # MS-SSIM
    if verbose:
        print("  Computing MS-SSIM...")
    mssim_arr = compute_ms_ssim(ssp_truth, ssp_ae, sample=10)
    metrics_dict["ms_ssim"] = float(np.mean(mssim_arr))
    if "MS_SSIM" in data_dict_metrics and mssim_arr.shape[0] == ssp_truth.shape[0]:  # Only compute best/worst if we have per-profile MS-SSIM values
        data["MS_SSIM"] = get_best_worst_random(mssim_arr, ssp_truth, ssp_ae, "MS_SSIM", best_is_min=False)
    del mssim_arr
    gc.collect()

    # # # NSR (effective resolution)
    # # if verbose:
    # #     print("\n" + "="*70)
    # #     print("  Computing NSR (spectral resolution)...")
    # #     print("="*70)
    # #     log_memory_checkpoint("Before NSR computation (original)", verbose=True)
    # #     log_data_shapes({
    # #         'ssp_truth_da': ssp_truth_da,
    # #         'ssp_ae_da': ssp_ae_da
    # #     }, "NSR Input Data Shapes (Original)")
    
    # # try:
    # #     for idx_t, t_r in enumerate([0.1,0.5,0.9], 1):
    # #         if verbose:
    # #             print(f"  [{idx_t}/3] Computing NSR with target_ratio={t_r}...")
    # #             log_memory_checkpoint(f"Before nsr_depth (t_r={t_r})", verbose=True)
            
    # #         nsr_depth = compute_nsr_along_depth(ssp_truth_da, ssp_ae_da, target_ratio=t_r)
            
    # #         if verbose:
    # #             log_memory_checkpoint(f"After nsr_depth (t_r={t_r})", verbose=True)
    # #             print(f"    nsr_depth result: {nsr_depth}")
            
    # #         nsr_map = compute_nsr_spatial_map(ssp_truth_da, ssp_ae_da, target_ratio=t_r)
            
    # #         if verbose:
    # #             log_memory_checkpoint(f"After nsr_map (t_r={t_r})", verbose=True)
    # #             print(f"    nsr_map result: {nsr_map}")
            
    # #         metrics_dict[f"nsr_depth_resolution_{int(t_r*100)}"] = float(nsr_depth["resolution"])
    # #         metrics_dict[f"nsr_spatial_resolution_{int(t_r*100)}"] = float(nsr_map["resolution"])
            
    # #         if verbose:
    # #             print(f"    ✓ Completed NSR for t_r={t_r}")
    # # except Exception as e:
    # #     if verbose:
    # #         print(f"  ✗ Error during NSR computation: {e}")
    # #         log_memory_checkpoint("After NSR error (original)", verbose=True)
    # #         import traceback
    # #         traceback.print_exc()
    # #     raise
    
    # # if verbose:
    # #     log_memory_checkpoint("After all NSR computations (original)", verbose=True)
    # #     gc.collect()
    # #     log_memory_checkpoint("After garbage collection (original)", verbose=True)


    # RMSE of pca reconstruction
    ae_pca = pca.transform(ssp_ae.transpose(0, 2, 3, 1).reshape(-1, ssp_ae.shape[1]))
    for n_components in [3,6]:
        ae_pca_n = ae_pca[:, :n_components]
        ssp_ae_pca_recon = (
            ae_pca_n @ pca.components_[:n_components, :] + pca.mean_
        )

        ssp_ae_pca_recon = ssp_ae_pca_recon.reshape(ssp_ae.shape[0], ssp_ae.shape[2], ssp_ae.shape[3], ssp_ae.shape[1]).transpose(0, 3, 1, 2)
        rmse_pca = float(np.sqrt(((ssp_ae_pca_recon - ssp_truth) ** 2).mean()))
        metrics_dict[f"rmse_pca_{n_components}_components"] = rmse_pca
        if f"RMSE_PCA" in data_dict_metrics:
            metric_arr = np.sqrt(((ssp_ae_pca_recon - ssp_truth) ** 2).mean(axis=1))
            data[f"RMSE_PCA"] = get_best_worst_random(metric_arr, ssp_truth, ssp_ae, f"RMSE_PCA_{n_components}_COMPONENTS", best_is_min=True)


    

    
    if verbose:
        print(f"\n✓ Computed {len(metrics_dict)} metric values")
    
    return metrics_dict , data


def compute_total_bits(out_net) -> float:
    return sum(torch.log(likelihoods).sum() / (-math.log(2)) for likelihoods in out_net['likelihoods'].values()).item()

def compute_bpe(out_net, num_pixels) -> float:
    return sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    ).item()



def parse_mlic_config(ckpt_path):
    """
    Parse experiment_config.log file to extract N and M values.
    
    Args:
        ckpt_path (str): PosixPath to checkpoint 
        
    Returns:
        tuple: (N, M) values or (None, None) if parsing fails
    """

    config_file = list((ckpt_path.parent.parent).rglob("train_*.log")) 
    if not config_file:
        print(f"Warning: train logs not found in {ckpt_path}")
        return None, None
    
    config_file = config_file[0]  # Get the first match if multiple found

    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Extract N and M values from MODEL CONFIGURATION section
        N, M = None, None
        
        lines = content.split('\n')

        cfg = eval((lines[1].strip().split("INFO: ")[-1]).replace("<class 'torch.nn.modules.activation.","nn.").replace("'>",""))


        return cfg
        
            
    except Exception as e:
        print(f"Error parsing config file in {ckpt_path}: {e}")
        return None


def find_checkpoints(checkpoint_paths: List[str]) -> List[Path]:
    """
    Find checkpoint files in specified paths.
    
    Parameters
    ----------
    checkpoint_paths : List[str]
        List of paths to search for checkpoints
        
    Returns
    -------
    List[Path]
        List of checkpoint file paths
    """
    checkpoints = []
    for path_str in checkpoint_paths:
        path = Path(path_str)
        if path.is_file():
            checkpoints.append(path)
        else:
            # Search for .tar files in directory
            #found = sorted(path.rglob("*f1*.tar"))  #+sorted(path.rglob("*rmse*.tar")) +  
            found = sorted(path.rglob("*.tar"))
            checkpoints.extend(found)
    
    return sorted(list(set(checkpoints)))


def process_model_in_batches(model, data_tensor, batch_size=4, dim=0, device='cuda'):
    """
    Process large tensors through model in batches to avoid OOM.
    
    Args:
        model: PyTorch model
        data_tensor: Input tensor
        batch_size: Number of samples per batch
        dim: Dimension along which to batch (0=time, 2=lat, 3=lon)
        device: Device to use
        
    Returns:
        numpy array of model outputs concatenated
    """
    results = []
    n_samples = data_tensor.shape[dim]
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
            end_idx = min(i + batch_size, n_samples)
            
            # Create slice for the batch
            slices = [slice(None)] * len(data_tensor.shape)
            slices[dim] = slice(i, end_idx)
            
            batch_data = data_tensor[tuple(slices)]
            batch_output = model(batch_data)
            results.append(batch_output)
    
    # Concatenate along the same dimension
    output = torch.cat(results, dim=dim)
    return output


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'cuda', dm=None, batch=None ) -> Tuple[torch.nn.Module, dict]:
    """
    Load model from checkpoint file.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
    device : str
        Device to load model on
        
    Returns
    -------
    Tuple[torch.nn.Module, dict]
        Loaded model and checkpoint state
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect model type from checkpoint
    if 'MLIC' in checkpoint_path:
        # MLIC format
        cfg = parse_mlic_config(Path(checkpoint_path))
        model = MLICPlusPlus(config=Config(cfg))
        model.load_state_dict(checkpoint['state_dict'])
        if model.in_channels == 3:
            model_type = "RGB_MLIC"
        else:
            model_type = "MLIC"
    elif 'AE' in checkpoint_path:
        # CAE format - assume it's directly the state dict or wrapped
        try:
            cfg = get_cfg_from_ckpt_path(str(checkpoint_path), pprint=False)
            model = load_model(checkpoint_path, dm, batch)
            model_type = "CAE"
        except:
            raise ValueError(f"Could not load model from {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint, model_type


def run_rgb_mlic(model, ssp_tensor):
    """
    Run RGB MLIC model by compressing and decompressing each channel separately.
    
    Args:
        model: MLIC model instance
        ssp_tensor: Input tensor of shape (N, 3, H, W)
    """

    reconstructed_tens = torch.zeros(ssp_tensor.shape,dtype=ssp_tensor.dtype).to(device=ssp_tensor.device)
    total_bits = 0
    total_elements = 0
    total_original_bits = 0
    n_imgs = ssp_tensor.shape[1] // 3
    for j in range(n_imgs):
        start_idx = j * 3
        end_idx = start_idx + 3

        x = torch.tensor(ssp_tensor[:, start_idx:end_idx, :, :]).to(device=ssp_tensor.device, dtype=ssp_tensor.dtype)
        with torch.no_grad():
            rv = model(x)
        bits = compute_total_bits(rv)
        numel = rv['x_hat'].numel()
        original_bits = numel * rv['x_hat'].element_size() * 8
        total_bits += bits
        total_elements += numel
        total_original_bits += original_bits
        reconstructed_tens[:, start_idx:end_idx] = rv['x_hat'].detach()
        if torch.mean(reconstructed_tens[:,-1])==0:
            reconstructed_tens[:,-1] = reconstructed_tens[:,-2]
        bpe = total_bits / total_elements if total_elements > 0 else 0
        cr = total_original_bits / total_bits if total_bits > 0 else float('inf')

    return reconstructed_tens, cr, bpe


def compute_metrics_for_checkpoints(
    checkpoint_paths: List[str],
    dm_path: str,
    data_dict_metrics: List[str],
    time_sample: int = 10,
    compute_pca: bool = False,
    norm_stats: str = "train",
    verbose: bool = False,
    unique_name: str = "",
) -> Dict[str, Dict]:
    """
    Compute metrics for one or more model checkpoints.
    
    Parameters
    ----------
    checkpoint_paths : List[str]
        List of checkpoint paths or directories to search
    dm_path : str
        Path to datamodule pickle file
    data_dict_metrics : List[str]
        List of metrics to compute
    compute_pca : bool, optional
        If True, compute PCA-based metrics
    verbose : bool, optional
        If True, print progress information
    unique_name : str, optionalfca
        Unique name for experiment
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping checkpoint names to computed metrics
    """
    import pickle
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_paths)
    if verbose:
        print(f"Found {len(checkpoints)} checkpoint(s)")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
    
    # Load datamodule
    if verbose:
        print(f"\nLoading datamodule from {dm_path}...")
    with open(dm_path, 'rb') as f:
        dm = pickle.load(f)
    
    # Get test data
    ssp_truth_da = dm.test_ds.input
    ssp_truth_da = ssp_truth_da.isel(time=slice(None, None, time_sample))  # Subsample for faster metric computation
    ssp_truth = ssp_truth_da.values.astype(np.float32)

    depth_array = ssp_truth_da.z.values
    dm_test_norm = ssp_truth_da.attrs['norm_stats']
    dm_train_norm = dm.train_ds.input.attrs['norm_stats']

    batch_tens = next(iter(dm.test_dataloader())).to(device)
    
    # Unnormalize
    ssp_truth_da = unorm_ssp_arr_3D(ssp_truth_da, dm_test_norm)

    crop_idx = slice(20, -20)
    ssp_truth_metrics_da = ssp_truth_da.isel(lat=crop_idx, lon=crop_idx)

    ssp_truth_metrics_grad_da = ssp_truth_metrics_da.differentiate("z")

    pca_train = PCA(n_components=15, svd_solver="randomized", random_state=42)
    train_truth_da = dm.train_ds.input
    train_truth_da = unorm_ssp_arr_3D(train_truth_da, train_truth_da.attrs['norm_stats']).isel(lat=crop_idx, lon=crop_idx)
    pca_train = pca_train.fit(train_truth_da.data.transpose(0, 2, 3, 1).reshape(-1, train_truth_da.shape[1]))


    pca_grad_train = PCA(n_components=15, svd_solver="randomized", random_state=42)
    grad_train_da = train_truth_da.differentiate("z")
    pca_grad_train = pca_grad_train.fit(grad_train_da.data.transpose(0, 2, 3, 1).reshape(-1, grad_train_da.shape[1]))

    del train_truth_da, grad_train_da
    

    
    if verbose:
        print(f"Test data shape: {ssp_truth_da.shape}")
    
    # Define seasons for seasonal filtering
    seasons = ['all'] #, 'spring', 'summer' 'autumn', 'winter'
    
    # Compute metrics for each checkpoint
    results = {}
    data_dict = {}
    
    for idx, checkpoint_path in tqdm(enumerate(checkpoints, 1), total=len(checkpoints), desc="Processing checkpoints"):
        ckpt_name = checkpoint_path.stem.split('.')[0]
 

        if verbose:
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(checkpoints)}] Processing: {ckpt_name}")
            print(f"{'='*70}")
        
        try:
            # Load model
            model, state, model_type = load_model_from_checkpoint(str(checkpoint_path), device=str(device), dm=dm, batch=batch_tens)
            
            if verbose:
                print(f"✓ Loaded {model_type} model")
            
            # Forward pass through model
            
            if norm_stats=="train":
                if "train_norm_stats" in state:
                    model_norm_stats = state["train_norm_stats"]
                else:
                    model_norm_stats = dm_train_norm
                    model_norm_stats["method"] = "mean_std_along_depth"

            elif norm_stats=="test":
                #model_norm_stats = state.get("test_norm_stats", dm_test_norm)
                if "test_norm_stats" in state:
                    model_norm_stats = state["test_norm_stats"]
                else:
                    model_norm_stats = dm_test_norm
                    model_norm_stats["method"] = "mean_std_along_depth"

            if model_type == "RGB_MLIC":
                model_norm_stats["method"] = "min_max"
            

            ssp_truth_norm = norm_ssp_arr_3D(ssp_truth_da, model_norm_stats).values.astype(np.float32)
            ssp_tensor = torch.from_numpy(ssp_truth_norm).to(device)
            original_size_bits = ssp_truth_norm.nbytes * 8


            with torch.no_grad():
                
                if model_type == "RGB_MLIC":
                    ssp_ae_tensor, cr, bpe = run_rgb_mlic(model, ssp_tensor)


                        
                elif model_type == "MLIC":
                    # output = model.compress(ssp_tensor)
                    # ssp_ae_tensor = model.decompress(output['strings'],output['shape'])['x_hat']
                    rv_batch = model(ssp_tensor)
                    ssp_ae_tensor = rv_batch['x_hat'].detach()
                    compressed_size_bits = compute_total_bits(rv_batch)
                    cr = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else float('inf')
                    N, _, H, W = ssp_tensor.size()
                    num_pixels = N * H * W
                    bpe = compute_bpe(rv_batch, num_pixels=num_pixels)
                
                                
                elif model_type == "CAE":
                    ssp_ae_tensor = process_model_in_batches(model, ssp_tensor, batch_size=4, dim=0, device=device)
                    #ssp_ae_tensor = model(ssp_tensor)
                    cr = model.model_AE.cr if hasattr(model.model_AE, 'cr') else 1.0
                    bpe = model.model_AE.bpe if hasattr(model.model_AE, 'bpe') else 0.0


            
            
            ssp_ae = ssp_ae_tensor.cpu().numpy().astype(np.float32)
            ssp_ae = unorm_ssp_arr_3D(ssp_ae, model_norm_stats)


            ssp_ae_da = xr.DataArray(
                ssp_ae,
                coords=ssp_truth_da.coords,
                dims=ssp_truth_da.dims,
                name='ssp_reconstructed'
            )
            

            # Crop and interpolate
            ssp_ae_da = ssp_ae_da.isel(lat=crop_idx, lon=crop_idx)

            grad_ae_da = ssp_ae_da.differentiate("z")

            
            # z_uniform = np.linspace(float(ssp_truth_da.z.min()), float(ssp_truth_da.z.max()), len(ssp_truth_da.z))
            # ssp_truth_da = ssp_truth_da.interp(z=z_uniform)

            
            # Compute metrics for each season
            for season in tqdm(seasons, desc="Processing seasons", disable=not verbose, leave=False):
                if verbose:
                    print(f"\n  [Metrics] Processing season: {season.upper()}")
                
                # Get seasonal indices
                season_indices = get_seasonal_time_indices(ssp_truth_metrics_da.time, season)
                
                if len(season_indices) == 0:
                    if verbose:
                        print(f"    ⚠ No data found for season: {season}")
                    continue
                
                # Select seasonal data
                ssp_truth_seasonal = ssp_truth_metrics_da.isel(time=season_indices)
                ssp_ae_seasonal = ssp_ae_da.isel(time=season_indices)
                grad_truth_seasonal = ssp_truth_metrics_grad_da.isel(time=season_indices)
                grad_ae_seasonal = grad_ae_da.isel(time=season_indices)
                
                if verbose:
                    print(f"    SSP shape: {ssp_ae_seasonal.shape}, Samples: {len(season_indices)}")
                    log_memory_checkpoint(f"Before {season.upper()} SSP metrics computation", verbose=True)
                
                # Compute SSP metrics for this season
                metrics, data = compute_metrics(
                    ssp_truth_seasonal,
                    ssp_ae_seasonal,
                    depth_array,
                    data_dict_metrics=data_dict_metrics,
                    pca=pca_train,
                    verbose=verbose
                )
                
                if verbose:
                    log_memory_checkpoint(f"After {season.upper()} SSP metrics computation", verbose=True)
                    print(f"    [Metrics] Processing gradient data (shape: {grad_ae_seasonal.shape})...")
                
                # # Compute gradient metrics for this season
                # metrics_grad, data_grad = compute_metrics(
                #     grad_truth_seasonal,
                #     grad_ae_seasonal,
                #     depth_array,
                #     data_dict_metrics=data_dict_metrics,
                #     pca=pca_grad_train,
                #     verbose=verbose
                # )
                metrics_grad, data_grad = {}, {}
                if verbose:
                    log_memory_checkpoint(f"After {season.upper()} gradient metrics computation", verbose=True)
                    gc.collect()
                    log_memory_checkpoint(f"After GC for {season.upper()}", verbose=True)
                
                # Add metadata
                metrics['model_type'] = model_type
                metrics['checkpoint_path'] = str(checkpoint_path)
                metrics['season'] = season
                metrics['num_samples'] = len(season_indices)
                
                metrics_grad['season'] = season
                metrics_grad['num_samples'] = len(season_indices)
                
                # Create model name
                if unique_name:
                    model_name = model_type
                else:
                    lst = str(checkpoint_path).split('/')
                    pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}|\d{8}_\d{6}"
                    date_str = next((x for x in lst if re.fullmatch(pattern, x)), None)
                    model_name = f"{model_type}_{date_str}_{ckpt_name}"
                
                # Store results with season key
                if model_name not in results:
                    results[model_name] = {}
                if cr not in results[model_name]:
                    results[model_name][cr] = {}
                
                results[model_name][cr][season] = {"SSP": metrics, "GRAD": metrics_grad}
                
                # Store data
                if model_name not in data_dict:
                    data_dict[model_name] = {}
                if cr not in data_dict[model_name]:
                    data_dict[model_name][cr] = {}
                
                data_dict[model_name][cr][season] = {"SSP": data, "GRAD": data_grad}
                
                if verbose:
                    print(f"  ✓ Successfully computed metrics for {season.upper()}")
            
            # Print summary of seasonal metrics
            if verbose:
                print(f"\n✓ Successfully computed metrics for all seasons: {ckpt_name}")
                print(f"\n{'Seasonal Metrics Summary':^70}")
                print("=" * 70)
                for season in seasons:
                    if model_name in results and cr in results[model_name] and season in results[model_name][cr]:
                        ssp_metrics = results[model_name][cr][season]["SSP"]
                        if ssp_metrics:
                            print(f"\n{season.upper()} (n={ssp_metrics.get('num_samples', 'N/A')} samples):")
                            print("-" * 70)
                            for key, value in sorted(ssp_metrics.items()):
                                if isinstance(value, (int, float)) and key not in ['checkpoint_path', 'num_samples', 'season']:
                                    if isinstance(value, float):
                                        print(f"  {key:.<45} {value:.6f}" if value < 100 else f"  {key:.<45} {value:.2e}")
                                    else:
                                        print(f"  {key:.<45} {value}")
        
        except Exception as e:
            if verbose:
                print(f"✗ Error processing {ckpt_name}: {str(e)}")
                import traceback
                traceback.print_exc()
            results[ckpt_name] = {'error': str(e), 'model_type': 'UNKNOWN'}


    if compute_pca:
        
        # Prepare training data for PCA fitting (unnormalized)
        train_ssp_unnorm = unorm_ssp_arr_3D(dm.train_ds.input.data.copy(), dm.train_ds.input.attrs['norm_stats'])
        train_ssp_flat = train_ssp_unnorm.transpose(0, 2, 3, 1).reshape(-1, train_ssp_unnorm.shape[1])
        
        # Prepare test data (already have metric_datatest as unnormalized)
        test_ssp_flat = ssp_truth.transpose(0, 2, 3, 1).reshape(-1, ssp_truth.shape[1])
        
        # Define hyperparameter grids
        n_components_list = [1, 2, 3, 4, 5, 10, 20, 50, 100, test_ssp_flat.shape[1]] # 
        n_layer_pooling_list = [0, 1, 2, 3, 4, 5, 6, 7, 8] 
        
        for n_components in tqdm(n_components_list, desc='PCA n_components', disable=not verbose):
            for n_layer_pooling in n_layer_pooling_list:
                try:
                    # Create model name
                    pca_model_name = f"PCA_nc{n_components}_npool{n_layer_pooling}" if not unique_name else f"PCA"

                                        
                    # Compute compression ratio
                    cr = (4 ** n_layer_pooling) * ssp_truth.shape[1] / n_components if n_components > 0 else float('inf')
                    
                    if verbose:
                        print(f"Processing: {pca_model_name}")
                    
                    # Fit PCA on training data
                    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
                    pca.fit(train_ssp_flat)
                    
                    # Transform test data
                    pca_ae = pca.transform(test_ssp_flat)
                    
                    # Apply pooling/upsampling
                    if n_layer_pooling > 0:
                        pca_ae_pooled = apply_pooling_upsample_to_pca(
                            pca_ae,
                            ssp_truth_shape=ssp_truth.shape,
                            n_layer_pooling=n_layer_pooling,
                            pooling_mode="mean"
                        )
                    else:
                        pca_ae_pooled = pca_ae
                    
                    # Inverse transform to get reconstruction
                    ssp_ae_flat = pca.inverse_transform(pca_ae_pooled)
                    ssp_ae = ssp_ae_flat.reshape(
                        ssp_truth.shape[0], 
                        ssp_truth.shape[2], 
                        ssp_truth.shape[3], 
                        ssp_truth.shape[1]
                    ).transpose(0, 3, 1, 2).astype(np.float32)

                    ssp_ae = unorm_ssp_arr_3D(ssp_ae, dm.test_ds.input.attrs['norm_stats'])


                    ssp_ae_da = xr.DataArray(
                        ssp_ae,
                        coords=ssp_truth_da.coords,
                        dims=ssp_truth_da.dims,
                        name='ssp_reconstructed'
                    )

                    # Crop and interpolate
                    ssp_ae_da = ssp_ae_da.isel(lat=crop_idx, lon=crop_idx)

                    grad_ae_da = ssp_ae_da.differentiate("z")
                   
                    # Compute metrics for each season
                    for season in tqdm(seasons, desc="Processing seasons (PCA)", disable=not verbose, leave=False):
                        if verbose:
                            print(f"    Processing season: {season.upper()}")
                        
                        # Get seasonal indices
                        season_indices = get_seasonal_time_indices(ssp_truth_metrics_da.time, season)
                        
                        if len(season_indices) == 0:
                            if verbose:
                                print(f"      ⚠ No data found for season: {season}")
                            continue
                        
                        # Select seasonal data
                        ssp_truth_seasonal = ssp_truth_metrics_da.isel(time=season_indices)
                        ssp_ae_seasonal = ssp_ae_da.isel(time=season_indices)
                        grad_truth_seasonal = ssp_truth_metrics_grad_da.isel(time=season_indices)
                        grad_ae_seasonal = grad_ae_da.isel(time=season_indices)
                        
                        metrics, data = compute_metrics(
                            ssp_truth_seasonal,
                            ssp_ae_seasonal,
                            depth_array,
                            data_dict_metrics=data_dict_metrics,
                            pca=pca,
                            verbose=verbose
                        )

                        metrics_grad, data_grad = compute_metrics(
                            grad_truth_seasonal,
                            grad_ae_seasonal,
                            depth_array,
                            data_dict_metrics=data_dict_metrics,
                            pca=pca_grad_train,
                            verbose=verbose
                        )
                        
                        # Compute metrics
                        if pca_model_name not in results:
                            results[pca_model_name] = {}
                        if cr not in results[pca_model_name]:
                            results[pca_model_name][cr] = {}
                        
                        results[pca_model_name][cr][season] = {"SSP": metrics, "GRAD": metrics_grad}
                        
                        # Store data
                        if pca_model_name not in data_dict:
                            data_dict[pca_model_name] = {}
                        if cr not in data_dict[pca_model_name]:
                            data_dict[pca_model_name][cr] = {}
                        
                        data_dict[pca_model_name][cr][season] = {"SSP": data, "GRAD": data_grad}


                except Exception as e:
                    if verbose:
                        print(f"  ✗ Error processing {pca_model_name}: {e}")
                    continue
    
    # Print summary table
    if verbose and results:
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY")
        print(f"{'='*70}")
        
        # Create summary DataFrame
        summary_rows = []
        for ckpt_name, metrics in results.items():
            row = {'checkpoint': ckpt_name, 'model_type': metrics.get('model_type', 'N/A')}
            if 'error' not in metrics:
                # Add key metrics
                row.update({
                    'RMSE': metrics.get('rmse_spatial_mean', np.nan),
                    'MAE': metrics.get('mae_spatial_mean', np.nan),
                    'R²': metrics.get('r2_score', np.nan),
                    'NSR_depth(m)': metrics.get('nsr_depth_resolution', np.nan),
                    'NSR_spatial(m)': metrics.get('nsr_spatial_resolution', np.nan),
                })
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
    
    return results,data_dict


if __name__ == "__main__":
    # print("Compute Metrics Module - Ready for use")
    # print("\nUsage example:")
    # print("  from compute_metrics import compute_metrics_for_checkpoints")
    # print("  results = compute_metrics_for_checkpoints(")
    # print("      checkpoint_paths=['/path/to/checkpoints/'],")
    # print("      dm_path='/path/to/datamodule.pkl',")
    # print("      data_dict_metrics=['RMSE', 'MAE', 'PSNR', 'NSR'],")
    # print("      compute_pca=True,")
    # print("      verbose=True")
    # print("  )")
    T0,LAT0,LON0 = 0, 50, 50

    #! to compute NSR, at min 300Gb of memory necessary 

    results,data_dict = compute_metrics_for_checkpoints(
        checkpoint_paths=["/Odyssey/private/o23gauvr/code/MLIC/experiments/mean_std_along_depth_complex_loss"], #, "/Odyssey/private/o23gauvr/code/MLIC/checkpoints/mlicpp_mse_q5_2960000.pth.tar"  #"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/eusipco/AE" "/Odyssey/private/o23gauvr/code/MLIC/checkpoints/mlicpp_mse_q5_2960000.pth.tar" #"/Odyssey/private/o23gauvr/code/MLIC/experiments/article_long" #, # # # Will rglob for *.tar
        dm_path='/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split_filtered_z_uniform_alternate_days_7_60_10.pkl',  #
        data_dict_metrics=[], #'RMSE', 'PEARSON', 'ECS', 'EXTREMUM_POS_ERROR', 'F1_SCORE', 'DTW', 'LSD', 'MS_SSIM'
        time_sample=1,
        compute_pca=False,
        norm_stats="train",
        verbose=True,
        unique_name=False
    )

    out_dir = Path('/Odyssey/private/o23gauvr/code/FASCINATION/pickle')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'model_metrics_mean_std_along_depth_complex_loss.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(out_dir / 'data_dict_mean_std_along_depth_complex_loss.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Saved model_metrics and data_dict to {out_dir}')

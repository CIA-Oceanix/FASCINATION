import os 
import sys
import gc

running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)



"""Compute model metrics and extract best/worst examples.

This module is a cleaned and refactored version of the previous notebook-derived
script. The algorithm is preserved. Use the CLI to set paths and device.
"""

from pathlib import Path
import argparse
import math
import pickle
import os
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from tqdm import tqdm
from scipy.ndimage import convolve
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from pytorch_msssim import ms_ssim
from sklearn.decomposition import PCA

import FASCINATION.src.utils as utils
from FASCINATION.src.utils import unorm_ssp_arr_3D
from FASCINATION.src.autoencoder_datamodule_natl_enatl import AEDatamodule, AE_BaseDataset_3D

from scipy.ndimage import gaussian_filter1d

from joblib import Parallel, delayed

try:
    from MLIC.MLIC.models import MLICPlusPlus
    from MLIC.MLIC.utils.utils import Config
except Exception:
    MLICPlusPlus = None
    Config = dict


def parse_experiment_config(ckpt_path):
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



def find_first_level_dirs(base_dir: str) -> Dict[str, str]:
    result = {}
    try:
        for name in next(os.walk(base_dir))[1]:
            if name == 'mute':
                continue
            result[name] = os.path.join(base_dir, name)
    except Exception:
        pass
    return result


def find_non_empty_checkpoint_dirs(base_dir: str, verbose: bool = False) -> Dict[str, str]:
    """Find all non-empty checkpoint directories recursively.
    
    Searches for directories containing .pth.tar checkpoint files.
    
    Args:
        base_dir (str): Base directory to search in
        verbose (bool): Enable verbose output
        
    Returns:
        Dict[str, str]: Dictionary mapping checkpoint folder names to their full paths
    """
    result = {}
    
    if not base_dir or not os.path.exists(base_dir):
        if verbose:
            print(f"Base directory does not exist: {base_dir}")
        return result
    
    try:
        # Recursively search for directories containing .pth.tar files
        for root, dirs, files in os.walk(base_dir):
            if "mute" in root:
                continue
            checkpoint_files = [f for f in files if f.endswith('.pth.tar')]
            if checkpoint_files:
                # Make a unique key for each checkpoint directory
                relative_path = os.path.relpath(root, base_dir)
                key = "_".join(root.split('/')[-4:-1])
                result[key] = root
                if verbose:
                    print(f"Found checkpoint dir: {key} -> {root} ({len(checkpoint_files)} files)")
    except Exception as e:
        print(f"Error scanning checkpoint directories: {e}")
    
    return result


def cubic_interpolate_along_axis(arr: np.ndarray, target_size: int, axis: int) -> np.ndarray:
    x_old = np.linspace(0, 1, arr.shape[axis])
    x_new = np.linspace(0, 1, target_size)
    arr_swapped = np.moveaxis(arr, axis, 0)
    reshaped = arr_swapped.reshape(arr_swapped.shape[0], -1)
    f = interp1d(x_old, reshaped, kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate')
    interpolated = f(x_new)
    new_shape = (target_size,) + arr_swapped.shape[1:]
    interpolated = interpolated.reshape(new_shape)
    return np.moveaxis(interpolated, 0, axis)


def get_min_max_idx(arr: np.ndarray, axs: int = 1, pad: bool = True) -> np.ndarray:
    grad = np.diff(arr, axis=axs)
    grad_sign = np.sign(grad)
    min_max = np.abs(np.sign(np.diff(grad_sign, axis=axs)))
    if pad:
        # pad for 4D arrays (t, z, lat, lon)
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axs] = (1, 1)
        min_max = np.pad(min_max, pad_width, 'constant', constant_values=1)
    return min_max


def get_f1_score(min_max_idx_truth: np.ndarray, min_max_idx_ae: np.ndarray, axs: int = 1, kernel_size: int = 10) -> np.ndarray:
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


def compute_psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:

    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')

    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def compute_msssim(a: torch.Tensor, b: torch.Tensor) -> float:
    return ms_ssim(a, b, data_range=b.max()-b.min()).item()


def compute_total_bits(out_net: Dict[str, torch.Tensor]) -> float:
    return sum(torch.log(likelihoods).sum() / (-math.log(2)) for likelihoods in out_net['likelihoods'].values()).item()


def ssim_1d(x, y, sigma=1.5, C1=1e-4, C2=1e-4):
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
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weights = weights[:scales]

    mssim = []
    x_s, y_s = x.copy(), y.copy()

    for i in range(scales):
        ssim_val = ssim_1d(x_s, y_s)
        mssim.append(ssim_val)

        # Downsample (low-pass + decimate)
        x_s = gaussian_filter1d(x_s, sigma=1)[::2]
        y_s = gaussian_filter1d(y_s, sigma=1)[::2]

    mssim = np.array(mssim)
    mssim = np.maximum(mssim, 0)

    return np.prod(mssim ** weights)




def compute_ms_ssim(ssp_truth: np.ndarray, ssp_ae: np.ndarray) -> np.ndarray:
    flatten_truth = ssp_truth.transpose(0,2,3,1).reshape(-1, ssp_truth.shape[1])
    flatten_ae = ssp_ae.transpose(0,2,3,1).reshape(-1, ssp_ae.shape[1])
    mssim_arr = []
    mssim_arr = Parallel(n_jobs=-1)(
    delayed(ms_ssim_1d)(flatten_truth[prof], flatten_ae[prof]) 
    for prof in tqdm(range(flatten_truth.shape[0]), mininterval=10.0, desc="Computing MS-SSIM for each profile"))
    mssim_arr = np.array(mssim_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    return mssim_arr


def apply_pooling_upsample_to_pca(pca_ae: np.ndarray, ssp_truth_shape: tuple, n_layer_pooling: int, pooling_mode: str = "mean") -> np.ndarray:
    """
    Apply spatial pooling and upsampling to PCA-transformed data using numpy/scipy, then inverse transform.
    Works entirely with numpy arrays using skimage.measure.block_reduce and scipy.interpolate.RectBivariateSpline.
    
    Args:
        pca_ae: flattened PCA-transformed array with shape (batch*lat*lon, n_components)
        ssp_truth_shape: tuple of original spatial shape (batch, depth, lat, lon)
        n_layer_pooling: number of pooling layers to apply (e.g., 3)
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
        batch_comps = []
        for c in range(n_components):
            data_2d = pca_ae_spatial[b, c, :, :]
            
            # Apply pooling layers
            for layer in range(n_layer_pooling):
                pool_size = 2
                if pooling_mode == "mean":
                    data_2d = block_reduce(data_2d, block_size=(pool_size, pool_size), func=np.mean)
                else:  # max pooling
                    data_2d = block_reduce(data_2d, block_size=(pool_size, pool_size), func=np.max)
            
            # Upsample back to original size using interpolation
            y_old = np.linspace(0, 1, data_2d.shape[0])
            x_old = np.linspace(0, 1, data_2d.shape[1])
            y_new = np.linspace(0, 1, lat)
            x_new = np.linspace(0, 1, lon)
            
            # Create 2D interpolation function
            spl = RectBivariateSpline(y_old, x_old, data_2d, kx=min(3, data_2d.shape[0]-1), ky=min(3, data_2d.shape[1]-1))
            upsampled = spl(y_new, x_new)
            batch_comps.append(upsampled)
        
        # Stack components: (n_components, lat, lon)
        batch_comps = np.array(batch_comps)
        pooled_upsampled_list.append(batch_comps)
    
    # Concatenate batches: (batch, n_components, lat, lon)
    pooled_upsampled_np = np.array(pooled_upsampled_list)
    
    # Reshape to flattened form (batch*lat*lon, n_components) for inverse transform
    pooled_upsampled_flat = pooled_upsampled_np.transpose(0, 2, 3, 1).reshape(-1, n_components)
    
    return pooled_upsampled_flat


def compute_metrics_for_arrays(test_ssp_truth: np.ndarray, ae_ssp_test: np.ndarray, depth_array: np.ndarray, verbose=False) -> Dict[str, float]:
    """Compute metrics for a single truth / reconstructed pair (both unnormalized).

    Returns a dict with the same keys used elsewhere in the script.
    """
    metrics = {}

    max_ssp_truth_idx = np.nanargmax(test_ssp_truth, axis=1)
    ecs_truth = depth_array[max_ssp_truth_idx]

    # # PSNR using truth range by default
    # truth_min = float(np.nanmin(test_ssp_truth))
    # truth_max = float(np.nanmax(test_ssp_truth))
    # truth_range = truth_max - truth_min if truth_max > truth_min else 1.0


    if verbose:        print("Computing PSNR...")
    psnr = compute_psnr(test_ssp_truth, ae_ssp_test)

    if verbose:        print("Computing MS-SSIM...")
    msssim = compute_ms_ssim(test_ssp_truth,ae_ssp_test).mean() 
    #msssim = np.nan
    # msssim = compute_msssim(torch.tensor(cubic_interpolate_along_axis(test_ssp_truth, 161, axis=2), dtype=torch.float64),
    #                         torch.tensor(cubic_interpolate_along_axis(ae_ssp_test, 161, axis=2), dtype=torch.float64))

    if verbose:        print("Computing ECS, RMSE, MAE")
    max_ssp_ae_idx = np.nanargmax(ae_ssp_test, axis=1)
    ecs_pred_ae = depth_array[max_ssp_ae_idx]

    ae_ssp_rmse = np.sqrt(np.mean((test_ssp_truth - ae_ssp_test) ** 2))
    ae_ecs_rmse = np.sqrt(np.mean((ecs_truth - ecs_pred_ae) ** 2))
    mae = np.mean(np.abs(test_ssp_truth - ae_ssp_test))

    if verbose:        print("Computing min/max indices and F1 score...")
    min_max_idx_truth = get_min_max_idx(test_ssp_truth, pad=False)
    min_max_idx_ae = get_min_max_idx(ae_ssp_test, pad=False)
    mean_error_n_min_max = np.mean(np.abs(np.sum(min_max_idx_truth, axis=1) - np.sum(min_max_idx_ae, axis=1)))

    F1_score = get_f1_score(min_max_idx_truth, min_max_idx_ae)
    f1_score = np.mean(F1_score)

    if verbose:        print("Computing R2 score...")
    r2 = 1 - (np.sum((test_ssp_truth - ae_ssp_test) ** 2) / np.sum((test_ssp_truth - np.mean(test_ssp_truth)) ** 2))

    # # Compute filtered F1 score
    # if verbose:        print("Computing Filtered F1 score...")
    # b, a = butter(N=2, Wn=0.107, btype='low', analog=False)
    # filtered_ae_test_arr = filtfilt(b, a, ae_ssp_test, axis=1)
    # min_max_idx_filtered_ae = get_min_max_idx(filtered_ae_test_arr, pad=False)
    # filtered_F1_score = get_f1_score(min_max_idx_truth, min_max_idx_filtered_ae)
    # filtered_f1_score = np.mean(filtered_F1_score)

    metrics = {
        'RMSE': ae_ssp_rmse,
        'PSNR': psnr,
        'MS-SSIM': msssim,
        'ECS': ae_ecs_rmse,
        'MAE': mae,
        'mean_error_n_min_max': mean_error_n_min_max,
        'F1_score': f1_score,
        #'Filtered_F1_score': filtered_f1_score,
        'R2_score': r2,
    }

    return metrics




def get_best_worst_random(test_ssp_truth_local, ae_ssp_test_local, depth_array_local, metric_name):
    if metric_name == 'RMSE':
        errors = np.sqrt(np.mean((test_ssp_truth_local - ae_ssp_test_local) ** 2, axis=1))
        score = -errors
    elif metric_name == 'MAE':
        errors = np.mean(np.abs(test_ssp_truth_local - ae_ssp_test_local), axis=1)
        score = -errors
    elif metric_name == 'ECS':
        max_truth_idx = np.nanargmax(test_ssp_truth_local, axis=1)
        max_pred_idx = np.nanargmax(ae_ssp_test_local, axis=1)
        ecs_truth = depth_array_local[max_truth_idx]
        ecs_pred = depth_array_local[max_pred_idx]
        errors = np.abs(ecs_truth - ecs_pred)
        score = -errors
    elif metric_name == 'R2':
        num = np.sum((test_ssp_truth_local - ae_ssp_test_local) ** 2, axis=1)
        den = np.sum((test_ssp_truth_local - np.mean(test_ssp_truth_local, axis=1, keepdims=True)) ** 2, axis=1)
        score = 1 - num / den
    elif metric_name == 'F1_score':
        min_max_idx_truth = get_min_max_idx(test_ssp_truth_local, pad=False)
        min_max_idx_ae = get_min_max_idx(ae_ssp_test_local, pad=False)
        score = get_f1_score(min_max_idx_truth, min_max_idx_ae)
    else:
        raise ValueError(f'Unknown metric {metric_name}')

    flat_score = score.reshape(-1)
    best_idx = np.argmax(flat_score)
    worst_idx = np.argmin(flat_score)

    def unravel(idx):
        return np.unravel_index(idx, score.shape)

    def extract(idx):
        t0, lat0, lon0 = idx
        return {
            metric_name: np.abs(score[t0, lat0, lon0]),
            't_lat': ((t0, lat0), test_ssp_truth_local[t0, :, lat0, :], ae_ssp_test_local[t0, :, lat0, :].copy()),  # .copy() to avoid keeping ref to full array
            't_lat_lon': ((t0, lat0, lon0), test_ssp_truth_local[t0, :, lat0, lon0], ae_ssp_test_local[t0, :, lat0, lon0].copy())
        }

    return {'best': extract(unravel(best_idx)), 'worst': extract(unravel(worst_idx))}


def compute_and_save(
    dm_pkl: str,
    mlic_base_dir: str,
    other_ckpt_base: str,
    out_pickle_dir: str,
    compute_pca: bool = False,
    device: str = None,
    verbose: bool = False,
    unique_name: bool = True,
    crop_slice: bool = 0,
    fuse_val_test: bool = False
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datamodules
    if verbose:
        print(f'Loading datamodule pickles: {dm_pkl}')
        with open(dm_pkl, 'rb') as f:
            dm = pickle.load(f)
    # with open(dm_cae_pkl, 'rb') as f:
    #     dm_cae = pickle.load(f)

    # dm_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_157_196_256_norm_per_split.pkl" #"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_157_196_256_good_split.pkl" #
    # with open(dm_path, 'rb') as f:
    #     dm_enatl = pickle.load(f)

    data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}
    


    depth_array = dm.depth_array

    #test_ssp_da = dm_cae.test_da #xr.open_dataarray(data_path['natl'])

    

    test_coords = dm.test_ds.input.attrs["original_space_coords"]

    if fuse_val_test:
        # val_coords = dm.val_ds.input.attrs["original_space_coords"]

        # test_coords = {
        #     'lat': np.concatenate([test_coords['lat'], val_coords['lat']]),
        #     'lon': np.concatenate([test_coords['lon'], val_coords['lon']])
        # }

        augmented_da = xr.concat([unorm_ssp_arr_3D(dm.test_ds.input, dm.test_ds.input.attrs['norm_stats']),
                                  unorm_ssp_arr_3D(dm.val_ds.input, dm.val_ds.input.attrs['norm_stats'])], dim='time')

                                  

    else:

    #metric_datatest = unorm_ssp_arr_3D(dm.test_ds.input.data, dm)
        
        augmented_da = dm.test_ds.input.copy()
        augmented_da[:] = unorm_ssp_arr_3D(augmented_da.data, dm.test_ds.input.attrs['norm_stats'])



    mlic_test_arr = augmented_da.data.copy()
    
    test_norm = augmented_da.attrs['norm_stats']
    season_idx = dm.test_ds.input.season_idx



    # train_months = np.unique(dm_enatl.train_ds.input.time.dt.month.values)
    # test_months = np.unique(dm_enatl.test_ds.input.time.dt.month.values)

    # if fuse_val_test:
    #     metric_datatest = xr.concat([
    #         unorm_ssp_arr_3D(dm.test_ds.input, dm.test_ds.input.attrs['norm_stats']),
    #         unorm_ssp_arr_3D(dm.val_ds.input, dm.val_ds.input.attrs['norm_stats'])
    #     ], dim='time')

    #     metric_datatest = metric_datatest.interp(
    #         lat=test_coords['lat'],
    #         lon=test_coords['lon'],
    #         method="nearest"
    #     ).data


    # else:
    #     metric_datatest = dm.test_ds.input.interp(
    #                 lat=test_coords['lat'],
    #                 lon=test_coords['lon'],
    #                 method="nearest"
    #                 )
    #     #metric_datatest = metric_datatest.where(metric_datatest.time.dt.month.isin(test_months), drop=True)
    #     metric_datatest = unorm_ssp_arr_3D(metric_datatest.data, test_norm)

    
    metric_datatest = unorm_ssp_arr_3D(dm.test_ds.input.data, test_norm)

    # cae_test_ds = dm_cae.test_ds.input
    # # Store unnormalized version for re-use in each checkpoint (will normalize fresh each time)
    # cae_test_ssp_arr_unnorm = cae_test_ds.data.copy()
    # cae_test_ssp_arr_unnorm = unorm_ssp_arr_3D(cae_test_ssp_arr_unnorm, cae_test_ds.attrs['norm_stats'])
    
    # train_time = dm_enatl.train_ds.input.time.values
    # test_time = dm_enatl.test_ds.input.time.values


    t_idx, lat_idx, lon_idx = 15, 101, 81

    batch_size = 4

    # x_min,x_max = dm.norm_stats["params"]['x_min'], dm.norm_stats['params']['x_max']
    # mean,std = dm_cae.norm_stats["params"].values()

    
    #metric_datatest = test_ssp_da.data
    


    # lat_size = len(test_ssp_da.lat)
    # lon_size = len(test_ssp_da.lon)
    # n_lat = int(np.ceil(lat_size / 64))
    # closest_lat_size = n_lat * 64
    # n_lon = int(np.ceil(lon_size / 64))
    # closest_lon_size = n_lon * 64
    # new_lat = np.linspace(test_ssp_da.lat.min().item(), test_ssp_da.lat.max().item(), closest_lat_size)
    # new_lon = np.linspace(test_ssp_da.lon.min().item(), test_ssp_da.lon.max().item(), closest_lon_size)
    # augmented_da = test_ssp_da.interp(lat=new_lat, lon=new_lon, method="cubic").astype(getattr(np, dm_cae.dtype_str))
    # mlic_test_arr = augmented_da.data

    # cae_test_da = (test_ssp_da - mean) / std
    # cae_test_tens = torch.tensor(cae_test_da.data, device=device, dtype=getattr(torch, dm_cae.dtype_str))
    
    
    # mlic_test_arr = (mlic_test_arr-x_min)/(x_max-x_min)
    # mlic_test_arr = mlic_test_arr.astype(getattr(np, dm.dtype_str))


    # data = "enatl"  # "natl" or "enatl"

    # sst_path = {"enatl": "/Odyssey/public/enatl60/sst/eNATL60-BLB002-SST-2009-2010-1_20.nc",
    #             "natl": "/Odyssey/public/natl60/sst/NATL60_IFREMER_sources_and_sst_2016_2017.nc"}

    # sst_da = xr.open_dataarray(sst_path[data])

    # sst_train = sst_da.interp_like(dm.train_dataloader().dataset.input, method="nearest").data
    # sst_test = sst_da.interp_like(dm.test_dataloader().dataset.input, method="nearest").data

    # # Compute mean and std from train SST
    # sst_mean = sst_train.mean()
    # sst_std = sst_train.std()

    # Normalize SST arrays
    #sst_train_norm = (sst_train - sst_mean) / sst_std
    #sst_test_norm = (sst_test - sst_mean) / sst_std
    if fuse_val_test:
        sst_test_norm = np.concatenate([
            dm.test_ds.input.attrs['sst'].data,
            dm.val_ds.input.attrs['sst'].data
        ], axis=0)
    
    else:
        sst_test_norm  = dm.test_ds.input.attrs['sst'].data

    # containers
    bit_rates = {}
    #outputs = {}
    
    #rgb_models = set()
    # containers for metrics and selected examples (avoid storing full arrays)
    model_metrics: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    data_dict: Dict[str, Dict[Any, Any]] = {}

    # --- MLIC checkpoints ---
    mlic_ckpt_dict = find_non_empty_checkpoint_dirs(mlic_base_dir, verbose=verbose) #find_first_level_dirs(mlic_base_dir)
    if verbose:
        print(f'Found {len(mlic_ckpt_dict)} first-level entries under {mlic_base_dir}')

    # Process SSP MLIC++ (in_channels==157)
    for i, (model_name, model_path) in tqdm(enumerate(mlic_ckpt_dict.items()), desc='SSP MLIC++ models', disable=not verbose):
        

        # ckpt_files = list(Path(model_path).rglob('checkpoint_best_loss.pth.tar'))
        # ckpt_files += list(Path(model_path).rglob('best_checkpoint_loss.pth.tar'))
        # ckpt_files += list(Path(model_path).rglob('best_checkpoint_rmse.pth.tar'))
        # ckpt_files += list(Path(model_path).rglob('best_checkpoint_ecs.pth.tar'))
        # ckpt_files += list(Path(model_path).rglob('best_checkpoint_f1.pth.tar'))
        # ckpt_files += list(Path(model_path).rglob('best_checkpoint_bpp_loss.pth.tar'))

        #ckpt_files = list(Path(model_path).rglob('best_checkpoint_f1.pth.tar'))
        ckpt_files = list(Path(model_path).rglob('best_checkpoint_ecs.pth.tar'))


        if not ckpt_files:
            continue


        for ckpt_file in ckpt_files:

            # if str(ckpt_file) != "/Odyssey/private/o23gauvr/code/FASCINATION/outputs/test/MLIC/fixed_weight_loss_64_96_1.0_CR_100000.0_good_split_enatl_mean_std_along_depth/20260114_185105/checkpoints/best_checkpoint_rmse.pth.tar":
            #     continue
                        
            if verbose:
                print(f'Inspecting SSP models in {model_name} -> {model_path}, ckpt: {ckpt_file}')


            if len(ckpt_files) > 1:
                sub_name = (str(ckpt_file).split(f'{model_name}/')[-1]).split('/')
                if "checkpoints" in sub_name:
                    sub_name.remove('checkpoints')
                sub_name = "_" + "_".join(sub_name).split(".pth")[0]
            else:
                sub_name = ""
            if unique_name:
                dict_model_name = 'MLIC++ SSP' 
            else:
                dict_model_name = (model_name.split("mlicpp_on_ssp_")[-1]).split("_min_max")[0] + sub_name
            


            cfg = parse_experiment_config(ckpt_file)
            if cfg.get('in_channels', None) == 3:
                continue

            if cfg.get('add_embedded_seasons', None) == True:
                cfg["add_seasons"] = {"use": True, "mode": "embed"}

            if "one_hot" in str(ckpt_file):
                cfg["add_seasons"] = {"use": True, "mode": "one_hot"}
                print("Using one_hot season encoding")

            N = cfg.get('N', 192)
            M = cfg.get('M', 320)
            ssp_config = Config({
                'N': N, 'M': M, 'slice_num': cfg.get('slice_num', 10),
                'context_window': cfg.get('context_window', 5), 'act': cfg.get('act', nn.GELU),
                'in_channels': cfg.get('in_channels', 157), 'out_channels': cfg.get('in_channels', 157), 
                "add_seasons": cfg.get("add_seasons", {"use": False, "mode": None}),
                "add_sst": cfg.get("add_sst", False),
                "enable_channel_context": cfg.get("enable_channel_context", True),
                "enable_local_context": cfg.get("enable_local_context", True),
                "enable_global_inter_context": cfg.get("enable_global_inter_context", True),
                "enable_global_intra_context": cfg.get("enable_global_intra_context", True)
            })
            

            mlic_test_arr = augmented_da.data.copy()  # augmented_da.where(augmented_da.time.dt.month.isin(test_months), drop=True).data.copy()
            #mlic_test_arr = unorm_ssp_arr_3D(mlic_test_arr, test_norm)




            try:
                net = MLICPlusPlus(config=ssp_config).eval().to(device)
                ck = torch.load(ckpt_file, map_location=device)
                if "one_hot_old" in str(ckpt_file) or str(ckpt_file) == "/Odyssey/private/o23gauvr/code/MLIC/experiments/full_loss_batch_4_add_season_hot_one_64_96_0.0018_min_max/20251001_001608/checkpoints/checkpoint_best_loss.pth.tar":
                    del ck['state_dict']["season_embed.weight"]
                net.load_state_dict(ck['state_dict'])


                test_norm = ck.get('train_norm_stats', dm.test_ds.input.attrs['norm_stats'])
                #test_norm_method =  test_norm['method'] #ck.get('train_norm_stats', dm.test_ds.input.attrs['norm_stats'])['method']


                if  test_norm['method'] == "min_max":
                    x_min = test_norm["params"]["x_min"].astype(mlic_test_arr.dtype)
                    x_max = test_norm["params"]["x_max"].astype(mlic_test_arr.dtype)    
                    mlic_test_arr = (mlic_test_arr - x_min) / (x_max - x_min)
                elif test_norm['method'] == "mean_std":
                    mean = test_norm['params']["mean"].astype(mlic_test_arr.dtype)
                    std = test_norm['params']["std"].astype(mlic_test_arr.dtype)
                    mlic_test_arr = (mlic_test_arr - mean) / std
                elif test_norm['method'] == "mean_std_along_depth":
                    mean = test_norm['params']["mean_along_depth"].astype(mlic_test_arr.dtype)
                    std = test_norm['params']["std_along_depth"].astype(mlic_test_arr.dtype)
                    mlic_test_arr = (mlic_test_arr - mean) / std


                #reconstructed_arr = np.zeros((mlic_test_arr.shape[0], mlic_test_arr.shape[1], len(test_coords['lat']), len(test_coords['lon'])))
                reconstructed_arr = np.zeros(mlic_test_arr.shape)
                n_batch = int(np.ceil(len(mlic_test_arr) / batch_size))
                total_bits = 0
                total_elements = 0
                total_original_bits = 0
                batch_metrics = []
                for b_idx in range(n_batch):
                    start_idx = b_idx * batch_size
                    end_idx = min((b_idx + 1) * batch_size, len(mlic_test_arr))
                    x_batch = torch.tensor(mlic_test_arr[start_idx:end_idx].copy()).to(device=device, dtype=getattr(torch, dm.dtype_str))
                    season_idx_batch = season_idx[start_idx:end_idx]
                    sst_test_norm_batch = sst_test_norm[start_idx:end_idx]
                    with torch.no_grad():
                        rv_batch = net(x_batch, season_idx_batch, sst_test_norm_batch)
                    
                    bits = compute_total_bits(rv_batch)
                    numel = rv_batch['x_hat'].numel()
                    original_bits = numel * rv_batch['x_hat'].element_size() * 8
                    total_bits += bits
                    total_elements += numel
                    total_original_bits += original_bits

                    # Unnormalize per batch
                    batch_recon = rv_batch['x_hat'].detach().cpu().numpy()

                    
                    # Free GPU memory immediately after copying to CPU
                    del x_batch, rv_batch
                    torch.cuda.empty_cache()


                    
                    batch_recon = utils.unorm_ssp_arr_3D(batch_recon, test_norm)
                    # Prepare batch-wise augmented_da
                    # batch_augmented_da = augmented_da.isel(time=slice(start_idx, end_idx)).copy()
                    # batch_augmented_da[:] = batch_recon
                    # # Interpolate per batch
                    # batch_test_ssp_arr = batch_augmented_da.interp(
                    #     lat=test_coords['lat'],
                    #     lon=test_coords['lon'],
                    #     method="nearest"
                    # ).data
                    
                    batch_test_ssp_arr = batch_recon

                    b, a = butter(N=2, Wn=0.107, btype='low', analog=False)
                    batch_test_ssp_arr = filtfilt(b, a, batch_test_ssp_arr, axis=1).astype(batch_recon.dtype)

                    reconstructed_arr[start_idx:end_idx] = batch_test_ssp_arr
                    # Compute metrics per batch
                    batch_truth = metric_datatest[start_idx:end_idx]

                    if crop_slice > 0:
                        batch_metrics.append(compute_metrics_for_arrays(batch_truth[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice], batch_test_ssp_arr[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice], depth_array))
                    else:
                        batch_metrics.append(compute_metrics_for_arrays(batch_truth, batch_test_ssp_arr, depth_array))
                    
                    # Free batch memory
                    del batch_recon, batch_test_ssp_arr, batch_truth  #batch_augmented_da
                
                # Free normalized input after inference loop
                del mlic_test_arr
                
                
                # Average metrics
                bpe = total_bits / total_elements
                cr = total_original_bits / total_bits

                bit_rates.setdefault(dict_model_name, {})[cr] = {
                    'bits_per_element': bpe, 'compression_rate': cr,
                    'total_compressed_bits': bits, 'total_original_bits': original_bits,
                    'model_config': f'N={N}, M={M}', 'original_model_name': model_name
                }
                if verbose:
                    print("Averaging metrics for model:", dict_model_name, " at CR:", cr)
                mean_metrics = {}
                for k in batch_metrics[0].keys():
                    vals = [bm[k] for bm in batch_metrics]
                    mean_metrics[k] = float(np.mean(vals))


                
                model_metrics.setdefault(dict_model_name, {})
                model_metrics[dict_model_name][cr] = mean_metrics


                #reconstructed_arr = rv['x_hat'].detach().cpu().numpy()

                #reconstructed_arr = utils.unorm_ssp_arr_3D(reconstructed_arr, dm)
                            
                # if "mean_std" in model_name:
                #     mean=dm.norm_stats['params']['mean']
                #     std=dm.norm_stats['params']['std']
                #     if "along_depth" in model_name:
                #         reconstructed_arr = (reconstructed_arr*std) + mean

                #     else:
                #         reconstructed_arr = (reconstructed_arr*std.mean()) + mean.mean()

                # else:
                #     x_min,x_max = dm.norm_stats['params']['x_min'],dm.norm_stats['params']['x_max'] #dm.norm_stats['params'].values()
                #     reconstructed_arr = (reconstructed_arr*(x_max-x_min)) + x_min

                # augmented_da[:] = reconstructed_arr

                # #test_ssp_arr = augmented_da.data.copy()
                # test_ssp_arr = augmented_da.interp(
                # lat=test_coords['lat'],
                # lon=test_coords['lon'],
                # method="nearest"
                # ).data
                
                # Unnormalize truth for metrics (keep as numpy array)
                

                data_dict.setdefault(dict_model_name, {})
                
                # metrics = compute_metrics_for_arrays(metric_datatest, test_ssp_arr, depth_array)
                # model_metrics[dict_model_name][cr] = metrics
                
                # selection over last batch
                if verbose:
                    print("Selecting best/worst examples for model:", dict_model_name, " at CR:", cr)
                data_dict[dict_model_name][cr] = {}
                for metric_name in ["RMSE", "MAE", "F1_score", "ECS", "R2"]:

                    if crop_slice > 0:
                        data_dict[dict_model_name][cr][metric_name] = get_best_worst_random(metric_datatest[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice], reconstructed_arr[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice], depth_array, metric_name)
                    else:
                        data_dict[dict_model_name][cr][metric_name] = get_best_worst_random(metric_datatest, reconstructed_arr, depth_array, metric_name)
                
                data_dict[dict_model_name][cr]['selected'] = {
                    't_lat': ((t_idx, lat_idx), metric_datatest[t_idx, :, lat_idx, :], reconstructed_arr[t_idx, :, lat_idx, :].copy()),  # .copy() to avoid keeping ref to full array
                    't_lat_lon': ((t_idx, lat_idx, lon_idx), metric_datatest[t_idx, :, lat_idx, lon_idx], reconstructed_arr[t_idx, :, lat_idx, lon_idx].copy())
                }

                # metrics_cut = compute_metrics_for_arrays(metric_datatest[:, :-1, ...], test_ssp_arr[:, :-1, ...], depth_array[:-1])
                # model_metrics.setdefault(f'cutted_{dict_model_name}', {})[cr] = metrics_cut
                
            except Exception as e:
                print(e)
            finally:
                # Memory cleanup after each checkpoint
                if 'net' in dir():
                    del net
                if 'ck' in dir():
                    del ck
                if 'reconstructed_arr' in dir():
                    del reconstructed_arr
                if 'rv_batch' in dir():
                    del rv_batch
                if 'batch_metrics' in dir():
                    del batch_metrics
                torch.cuda.empty_cache()
                gc.collect()

    if verbose:
        print("Finished SSP MLIC++ models processing.")
    # Process RGB MLIC++ (in_channels==3)

    test_norm = augmented_da.attrs['norm_stats']
    mlic_test_arr = dm.test_ds.input.data.copy()
    mlic_test_arr = unorm_ssp_arr_3D(mlic_test_arr, test_norm)
    x_min = test_norm["params"]["x_min"]
    x_max = test_norm["params"]["x_max"]
    mlic_test_arr = (mlic_test_arr - x_min) / (x_max - x_min)
    test_norm["method"] = "min_max" # Ensure we know it's min_max for later unnormalization

        
    for i, (model_name, model_path) in tqdm(enumerate(mlic_ckpt_dict.items()), desc='RGB MLIC++ models', disable=not verbose):
        
        # if i>2:
        #     break

        ckpt_files = list(Path(model_path).rglob('best_checkpoint_loss.pth.tar'))
        #ckpt_files += list(Path(model_path).rglob('best_checkpoint_loss.pth'))
        if not ckpt_files:
            continue
        for ckpt_file in ckpt_files:
            if verbose:
                print(f'Inspecting RGB models in {model_name} -> {model_path}, ckpt: {ckpt_file}')
            cfg = parse_experiment_config(ckpt_file)
            if cfg.get('in_channels', None) != 3:
                continue
            N = cfg.get('N', 192)
            M = cfg.get('M', 320)
            ssp_config = Config({'N': N, 'M': M, 'slice_num': cfg.get('slice_num', 10), 'context_window': cfg.get('context_window', 5), 'act': cfg.get('act', nn.GELU), 'in_channels': 3, 'out_channels': 3})
            try:
                net = MLICPlusPlus(config=ssp_config).eval().to(device)
                ck = torch.load(ckpt_file, map_location=device)
                net.load_state_dict(ck['state_dict'])
                reconstructed_arr = np.zeros(mlic_test_arr.shape)
                total_bits = 0
                total_elements = 0
                total_original_bits = 0
                n_imgs = len(depth_array) // 3
                for j in range(n_imgs):
                    start_idx = j * 3
                    end_idx = start_idx + 3
                    x = torch.tensor(mlic_test_arr[:, start_idx:end_idx, :, :].copy()).to(device, dtype=getattr(torch, dm.dtype_str))
                    with torch.no_grad():
                        rv = net(x,season_idx,sst_test_norm)
                    bits = compute_total_bits(rv)
                    numel = rv['x_hat'].numel()
                    original_bits = numel * 8
                    total_bits += bits
                    total_elements += numel
                    total_original_bits += original_bits
                    reconstructed_arr[:, start_idx:end_idx] = rv['x_hat'].detach().cpu().numpy()

                if np.mean(reconstructed_arr[:,-1])==0:
                    reconstructed_arr[:,-1] = reconstructed_arr[:,-2]
                bpe = total_bits / total_elements
                cr = total_original_bits / total_bits
                if len(ckpt_files) > 1:
                    sub_name = "_" + (str(ckpt_file).split(f'{model_name}/')[-1]).split('/')[0]
                else:
                    sub_name = ""
                if unique_name:
                    dict_model_name = 'MLIC++ RGB' 
                else:
                    dict_model_name = model_name.split('_min_max')[0] + sub_name
                bit_rates.setdefault(dict_model_name, {})[cr] = {'bits_per_element': bpe, 'compression_rate': cr, 'total_compressed_bits': total_bits, 'total_original_bits': total_original_bits, 'model_config': f'N={N}, M={M}', 'original_model_name': model_name}
                
                reconstructed_arr = utils.unorm_ssp_arr_3D(reconstructed_arr, test_norm)
                #augmented_da[:] = reconstructed_arr
                ## Interpolate per batch
                # test_ssp_arr = augmented_da.interp(
                #     lat=test_coords['lat'],
                #     lon=test_coords['lon'],
                #     method="nearest"
                # ).data

                test_ssp_arr = reconstructed_arr

                b, a = butter(N=2, Wn=0.107, btype='low', analog=False)
                test_ssp_arr = filtfilt(b, a, test_ssp_arr, axis=1).astype(metric_datatest.dtype)


                model_metrics.setdefault(dict_model_name, {})
                data_dict.setdefault(dict_model_name, {})
                metrics = compute_metrics_for_arrays(metric_datatest, test_ssp_arr, depth_array)
                model_metrics[dict_model_name][cr] = metrics
                # selection
                data_dict[dict_model_name][cr] = {}
                for metric_name in ["RMSE", "MAE", "F1_score", "ECS", "R2"]:
                    data_dict[dict_model_name][cr][metric_name] = get_best_worst_random(metric_datatest, test_ssp_arr, depth_array, metric_name)
                
                data_dict[dict_model_name][cr]['selected'] = {
                    't_lat': ((t_idx, lat_idx), metric_datatest[t_idx, :, lat_idx, :], test_ssp_arr[t_idx, :, lat_idx, :].copy()),
                    't_lat_lon': ((t_idx, lat_idx, lon_idx), metric_datatest[t_idx, :, lat_idx, lon_idx], test_ssp_arr[t_idx, :, lat_idx, lon_idx].copy())
                }

                # metrics_cut = compute_metrics_for_arrays(metric_datatest[:, :-1, ...], test_ssp_arr[:, :-1, ...], depth_array[:-1])
                # model_metrics.setdefault(f'cutted_{dict_model_name}', {})[cr] = metrics_cut

            except Exception as e:
                print(e)

    # --- User-provided original MLICC checkpoint ---
    # This block loads the specific original MLIC checkpoint the user asked for,
    # runs inference in the same triple-channel loop as the RGB MLIC above,
    # and registers the outputs and bitrate info under the name 'original_mlicc'.
    orig_mlicc_ckpt = "" #"/Odyssey/private/o23gauvr/code/MLIC/checkpoints/mlicpp_mse_q5_2960000.pth.tar"
    if MLICPlusPlus is not None and Path(orig_mlicc_ckpt).exists() and orig_mlicc_ckpt != "":
        
        if verbose:
            print(f'Loading original MLICC checkpoint from {orig_mlicc_ckpt}')
        try:
            mlic_config_rgb = Config({
                "N": 192,
                "M": 320,
                "slice_num": 10,
                "context_window": 5,
                "act": nn.GELU,
                "in_channels": 3,
                "out_channels": 3
            })
            rgb_mlic_net = MLICPlusPlus(config=mlic_config_rgb).eval().to(device)
            ck = torch.load(orig_mlicc_ckpt, map_location=device)
            # support either a dict with 'state_dict' or a raw state dict
            state = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
            rgb_mlic_net.load_state_dict(state)

            reconstructed_arr = np.zeros(mlic_test_arr.shape,dtype=mlic_test_arr.dtype)
            total_bits = 0
            total_elements = 0
            total_original_bits = 0
            n_imgs = len(depth_array) // 3
            for j in range(n_imgs):
                start_idx = j * 3
                end_idx = start_idx + 3

                x = torch.tensor(mlic_test_arr[:, start_idx:end_idx, :, :].copy()).to(device=device, dtype=getattr(torch, dm.dtype_str))
                with torch.no_grad():
                    rv = rgb_mlic_net(x)
                bits = compute_total_bits(rv)
                numel = rv['x_hat'].numel()
                original_bits = numel * 8
                total_bits += bits
                total_elements += numel
                total_original_bits += original_bits
                reconstructed_arr[:, start_idx:end_idx] = rv['x_hat'].detach().cpu().numpy()
            if np.mean(reconstructed_arr[:,-1])==0:
                reconstructed_arr[:,-1] = reconstructed_arr[:,-2]
            bpe = total_bits / total_elements if total_elements > 0 else 0
            cr = total_original_bits / total_bits if total_bits > 0 else float('inf')
            dict_model_name = 'MLIC++ original' if unique_name else 'original_mlicc'
            bit_rates.setdefault(dict_model_name, {})[cr] = {
                'bits_per_element': bpe,
                'compression_rate': cr,
                'total_compressed_bits': total_bits,
                'total_original_bits': total_original_bits,
                'model_config': f'N=192, M=320',
                'original_model_name': orig_mlicc_ckpt
            }

            

            reconstructed_arr=unorm_ssp_arr_3D(reconstructed_arr, test_norm)
            # augmented_da[:] = reconstructed_arr
            
            # test_ssp_arr = augmented_da.interp(
            # lat=test_coords['lat'],
            # lon=test_coords['lon'],
            # method="nearest"
            # ).data

            test_ssp_arr = reconstructed_arr

            b, a = butter(N=2, Wn=0.107, btype='low', analog=False)
            test_ssp_arr = filtfilt(b, a, test_ssp_arr, axis=1).astype(metric_datatest.dtype)


            model_metrics.setdefault(dict_model_name, {})
            data_dict.setdefault(dict_model_name, {})
            metrics = compute_metrics_for_arrays(metric_datatest, test_ssp_arr, depth_array)
            model_metrics[dict_model_name][cr] = metrics
            # selection
            data_dict[dict_model_name][cr] = {}
            for metric_name in ['RMSE', 'MAE', 'F1_score', 'ECS', 'R2']:
                data_dict[dict_model_name][cr][metric_name] = get_best_worst_random(metric_datatest, test_ssp_arr, depth_array, metric_name)
            
            data_dict[dict_model_name][cr]['selected'] = {
                't_lat': ((t_idx, lat_idx), metric_datatest[t_idx, :, lat_idx, :], test_ssp_arr[t_idx, :, lat_idx, :].copy()),
                't_lat_lon': ((t_idx, lat_idx, lon_idx), metric_datatest[t_idx, :, lat_idx, lon_idx], test_ssp_arr[t_idx, :, lat_idx, lon_idx].copy())
            }

            # metrics_cut = compute_metrics_for_arrays(metric_datatest[:, :-1, ...], test_ssp_arr[:, :-1, ...], depth_array[:-1])
            # model_metrics.setdefault(f'cutted_{dict_model_name}', {})[cr] = metrics_cut

        except Exception as e:
            if verbose:
                print(f'  Could not load/run original_mlicc from {orig_mlicc_ckpt}: {e}')

    # --- Other models loaded from ckpt directories ---
    ckpt_dict = find_first_level_dirs(other_ckpt_base)

    

    if verbose:
        print(f'Found {len(ckpt_dict)} model directories under {other_ckpt_base}')
    for i, model_carac in tqdm(enumerate(ckpt_dict.keys()), desc='Models', disable=not verbose):
        

        #ckpt_list = list(Path(ckpt_dict[model_carac]).rglob('*.ckpt'))
        ckpt_list = list(Path(ckpt_dict[model_carac]).rglob('*.tar'))
        #ckpt_list = list(Path(ckpt_dict[model_carac]).rglob('best_checkpoint_f1.pth.tar'))
        #ckpt_list = list(Path(ckpt_dict[model_carac]).rglob('best_checkpoint_rmse.pth.tar'))
        
        target_name = 'CAE' if unique_name else model_carac
        #bit_rates.setdefault(target_name, {})
        for ckpt_path in ckpt_list:
            if unique_name:
                target_name = 'CAE'
            else:
                target_name = str(ckpt_path).split('/')[-3] + "_" + str(ckpt_path).split('/')[-1]
            bit_rates.setdefault(target_name, {})
            if verbose:
                print(f'Loading model checkpoint: {ckpt_path}')
            try:
                cfg = utils.get_cfg_from_ckpt_path(str(ckpt_path), pprint=False)
            except Exception:
                if verbose:
                    print(f'Could not get config for {ckpt_path}, skipping')
                continue

            try:
                ck = torch.load(ckpt_path, map_location=device)
                test_norm = ck.get('test_norm_stats', None)
                test_norm_method = test_norm['method']
            except Exception as e:
                if verbose:
                    print(f'Could not get normalization stats for {ckpt_path}: {e}')
                continue
                # test_norm = augmented_da.attrs['norm_stats'].copy()
                # test_norm_method = "mean_std_along_depth"

            # Create fresh normalized copy for this checkpoint (avoid modifying shared array)
            cae_test_ssp_arr =  augmented_da.data.copy()  #cae_test_ssp_arr_unnorm.copy()
            if test_norm_method == "min_max":
                x_min = test_norm["params"]["x_min"].astype(cae_test_ssp_arr.dtype)
                x_max = test_norm["params"]["x_max"].astype(cae_test_ssp_arr.dtype)
                cae_test_ssp_arr = (cae_test_ssp_arr - x_min) / (x_max - x_min)
            elif test_norm_method == "mean_std":
                mean = test_norm['params']["mean"].astype(cae_test_ssp_arr.dtype)
                std = test_norm['params']["std"].astype(cae_test_ssp_arr.dtype)
                cae_test_ssp_arr = (cae_test_ssp_arr - mean) / std
            elif test_norm_method == "mean_std_along_depth":
                mean = test_norm['params']["mean_along_depth"].astype(cae_test_ssp_arr.dtype)
                std = test_norm['params']["std_along_depth"].astype(cae_test_ssp_arr.dtype)
                cae_test_ssp_arr = (cae_test_ssp_arr - mean) / std

            # Load model using a single sample to initialize
            sample_tens = torch.tensor(cae_test_ssp_arr[:1], device=device, dtype=getattr(torch, dm.dtype_str))
            
            try:
                lit_model = utils.load_model(str(ckpt_path), dm, sample_tens, verbose=False)
            except Exception as e:
                if verbose:
                    print(f'Could not load model {ckpt_path}, skipping: {e}')
                continue

            # Run model and compute metrics over batches
            ssp_ae_test_arr = np.zeros_like(cae_test_ssp_arr)
            n_batch = int(np.ceil(len(cae_test_ssp_arr) / batch_size))
            total_bits = 0
            total_elements = 0
            batch_metrics = []

            for b_idx in range(n_batch):
                start_idx = b_idx * batch_size
                end_idx = min((b_idx + 1) * batch_size, len(cae_test_ssp_arr))
                x_batch = torch.tensor(cae_test_ssp_arr[start_idx:end_idx].copy(), device=device, dtype=getattr(torch, dm.dtype_str))
                
                with torch.no_grad():
                    batch_output = lit_model(x_batch)
                
                batch_recon = batch_output.detach().cpu().numpy().astype(getattr(np, dm.dtype_str))
                
                # Free GPU memory immediately
                del x_batch, batch_output
                torch.cuda.empty_cache()
                
                # Unnormalize per batch if needed
                if dm.norm_stats.get('norm_location', 'datamodule') == 'datamodule':
                    batch_recon = unorm_ssp_arr_3D(batch_recon, test_norm)
                
                b, a = butter(N=2, Wn=0.16, btype='low', analog=False)
                batch_recon = filtfilt(b, a, batch_recon, axis=1).astype(metric_datatest.dtype)

                ssp_ae_test_arr[start_idx:end_idx] = batch_recon
                
                # Accumulate bits info from model
                if hasattr(lit_model.model_AE, 'total_bits'):
                    total_bits += lit_model.model_AE.total_bits
                total_elements += batch_recon.size

                # Compute metrics per batch
                batch_truth = metric_datatest[start_idx:end_idx]
                batch_metrics.append(compute_metrics_for_arrays(batch_truth, batch_recon, depth_array))
                
                # Free batch memory
                del batch_recon, batch_truth
            
            # Free sample tensor used for initialization
            del sample_tens

            # Get compression info from model
            cr = lit_model.model_AE.cr if hasattr(lit_model.model_AE, 'cr') else 1.0
            bpe = lit_model.model_AE.bpe if hasattr(lit_model.model_AE, 'bpe') else 0.0
            if total_bits == 0 and hasattr(lit_model.model_AE, 'total_bits'):
                total_bits = lit_model.model_AE.total_bits

            # store bit rate info
            bit_rates[target_name][cr] = {'bits_per_element': bpe, 'compression_rate': cr, 'total_compressed_bits': total_bits}
            
            # Average metrics across batches
            try:
                model_metrics.setdefault(target_name, {})
                data_dict.setdefault(target_name, {})
                
                mean_metrics = {"full_path": str(ckpt_path)}
                for k in batch_metrics[0].keys():
                    vals = [bm[k] for bm in batch_metrics]
                    mean_metrics[k] = float(np.mean(vals))
                
                model_metrics[target_name][cr] = mean_metrics
                
                # selection
                data_dict[target_name][cr] = {}
                for metric_name in ['RMSE', 'MAE', 'F1_score', 'ECS', 'R2']: #'RMSE', 'F1_score', 'ECS', 'R2_score'
                    data_dict[target_name][cr][metric_name] = get_best_worst_random(metric_datatest, ssp_ae_test_arr, depth_array, metric_name)
                t_idx, lat_idx, lon_idx = 15, 101, 81
                data_dict[target_name][cr]['selected'] = {
                    't_lat': ((t_idx, lat_idx), metric_datatest[t_idx, :, lat_idx, :], ssp_ae_test_arr[t_idx, :, lat_idx, :].copy()),  # .copy() to avoid keeping ref
                    't_lat_lon': ((t_idx, lat_idx, lon_idx), metric_datatest[t_idx, :, lat_idx, lon_idx], ssp_ae_test_arr[t_idx, :, lat_idx, lon_idx].copy()),
                    'full_path': str(ckpt_path)
                }

                # metrics_cut = compute_metrics_for_arrays(metric_datatest[:, :-1, ...], ssp_ae_test_arr[:, :-1, ...], depth_array[:-1])
                # model_metrics.setdefault(f'cutted_{target_name}', {})[cr] = metrics_cut

            except Exception as e:
                if verbose:
                    print(f'Failed to compute metrics/selection for CAE {ckpt_path}: {e}')
            finally:
                # Memory cleanup after each CAE checkpoint
                if 'lit_model' in dir():
                    del lit_model
                if 'ssp_ae_test_arr' in dir():
                    del ssp_ae_test_arr
                if 'batch_metrics' in dir():
                    del batch_metrics
                if 'cae_test_ssp_arr' in dir():
                    del cae_test_ssp_arr
                if 'ck' in dir():
                    del ck
                torch.cuda.empty_cache()
                gc.collect()

    # --- PCA models with different n_components and n_layer_pooling ---
    if compute_pca:
        if verbose:
            print("Processing PCA models with different configurations...")
        
        # Prepare training data for PCA fitting (unnormalized)
        train_ssp_unnorm = unorm_ssp_arr_3D(dm.train_ds.input.data.copy(), dm.train_ds.input.attrs['norm_stats'])
        train_ssp_flat = train_ssp_unnorm.transpose(0, 2, 3, 1).reshape(-1, train_ssp_unnorm.shape[1])
        
        # Prepare test data (already have metric_datatest as unnormalized)
        test_ssp_flat = metric_datatest.transpose(0, 2, 3, 1).reshape(-1, metric_datatest.shape[1])
        
        # Define hyperparameter grids
        n_components_list = [1, 2, 3, 4, 5, 10, 20, 50, 100, metric_datatest.shape[1]] 
        n_layer_pooling_list = [0, 1, 2, 3, 4, 5, 6, 7, 8] 
        
        for n_components in tqdm(n_components_list, desc='PCA n_components', disable=not verbose):
            for n_layer_pooling in n_layer_pooling_list:
                try:
                    # Create model name
                    pca_model_name = f"PCA_nc{n_components}_npool{n_layer_pooling}" if not unique_name else f"PCA"
                    
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
                            ssp_truth_shape=metric_datatest.shape,
                            n_layer_pooling=n_layer_pooling,
                            pooling_mode="mean"
                        )
                    else:
                        pca_ae_pooled = pca_ae
                    
                    # Inverse transform to get reconstruction
                    ssp_ae_flat = pca.inverse_transform(pca_ae_pooled)
                    ssp_ae = ssp_ae_flat.reshape(
                        metric_datatest.shape[0], 
                        metric_datatest.shape[2], 
                        metric_datatest.shape[3], 
                        metric_datatest.shape[1]
                    ).transpose(0, 3, 1, 2)
                    
                    # Compute compression ratio
                    cr = (4 ** n_layer_pooling) * metric_datatest.shape[1] / n_components if n_components > 0 else float('inf')
                    
                    
                    # Compute metrics
                    model_metrics.setdefault(pca_model_name, {})
                    data_dict.setdefault(pca_model_name, {})
                    if crop_slice > 0:
                        metrics = compute_metrics_for_arrays(
                            metric_datatest[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice],
                            ssp_ae[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice],
                            depth_array
                        )
                    else:
                        metrics = compute_metrics_for_arrays(metric_datatest, ssp_ae, depth_array)
                    model_metrics[pca_model_name][cr] = metrics
                    
                    # Selection
                    data_dict[pca_model_name][cr] = {}
                    for metric_name in ['RMSE', 'MAE', 'F1_score', 'ECS', 'R2']:

                        if crop_slice > 0:
                            data_dict[pca_model_name][cr][metric_name] = get_best_worst_random(
                                metric_datatest[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice],
                                ssp_ae[:, :, crop_slice:-crop_slice, crop_slice:-crop_slice],
                                depth_array,
                                metric_name
                            )
                        else:
                            data_dict[pca_model_name][cr][metric_name] = get_best_worst_random(
                                metric_datatest, ssp_ae, depth_array, metric_name
                            )
                    
                    # Store selected examples
                    t_idx, lat_idx, lon_idx = 15, 101, 81
                    data_dict[pca_model_name][cr]['selected'] = {
                        't_lat': ((t_idx, lat_idx), metric_datatest[t_idx, :, lat_idx, :], ssp_ae[t_idx, :, lat_idx, :].copy()),
                        't_lat_lon': ((t_idx, lat_idx, lon_idx), metric_datatest[t_idx, :, lat_idx, lon_idx], ssp_ae[t_idx, :, lat_idx, lon_idx].copy()),
                        'config': f'n_components={n_components}, n_layer_pooling={n_layer_pooling}'
                    }
                    
                    if verbose:
                        print(f"  ✓ {pca_model_name}: CR={cr:.2f}, RMSE={metrics['RMSE']:.4f}")
                    
                    # Memory cleanup
                    del pca, pca_ae, pca_ae_pooled, ssp_ae_flat, ssp_ae
                    gc.collect()
                    
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Error processing {pca_model_name}: {e}")
                    continue
        
        # Cleanup training data
        del train_ssp_unnorm, train_ssp_flat, test_ssp_flat

    # (selection helper already defined above and used for CAE and MLIC models)

    # interpolate outputs to original_da resolution if needed and compute metrics on-the-fly
    # truth already unnormalized in `test_ssp_arr`

    # if verbose:
    #     print('Interpolating outputs to original datamodule resolution (if needed) and computing metrics')



    # Save outputs
    out_dir = Path(out_pickle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'model_metrics_test_cae_test_norm.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)
    with open(out_dir / 'data_dict_test_cae_test_norm.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Saved model_metrics and data_dict to {out_dir}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dm-pkl', default="/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split.pkl") #enatl_dm_157_196_256_good_split #  #"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_157_196_256_good_split.pkl")
    p.add_argument('--mlic-base-dir', default="") #/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/ICUA/MLIC # #/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/ICUA/MLIC #/Odyssey/private/o23gauvr/code/MLIC/experiments/ #/Odyssey/private/o23gauvr/code/MLIC/experiments #"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/eusipco/MLIC/")#/Odyssey/private/o23gauvr/code/FASCINATION/outputs/eusipco/MLIC/    #/Odyssey/private/o23gauvr/code/FASCINATION/outputs/test/MLIC #"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/MLIC++/icassp") #"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/MLIC++/icassp")#'/Odyssey/private/o23gauvr/code/MLIC/experiments') #'/Odyssey/private/o23gauvr/code/MLIC/experiments')  #'/Odyssey/private/o23gauvr/code/MLIC/experiments')#'/Odyssey/private/o23gauvr/code/MLIC/experiments/keep')
    p.add_argument('--other-ckpt-base', default="/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/AE_CNN/cr_1000") #/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/eusipco/AE # #"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/eusipco/AE") #/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/eusipco/AE #"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/CAE") #'/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/CAE') #'/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/CAE' #"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/CAE_visu_icassp")#
    p.add_argument('--out-pickle-dir', default='/Odyssey/private/o23gauvr/code/FASCINATION/pickle')
    p.add_argument('--compute-pca', action='store_true', default=False, help='Whether to compute PCA models with various n_components and pooling')
    p.add_argument('--device', default='cuda')
    p.add_argument('--verbose', action='store_true', default=True, help='Enable verbose prints and progress bars')
    p.add_argument('--unique-name', action='store_true', default=False, help='Use unique names for MLIC++ models (no hyperparam details)')
    return p.parse_args()


def main():
    args = parse_args()
    compute_and_save(
        args.dm_pkl,
        args.mlic_base_dir,
        args.other_ckpt_base,
        args.out_pickle_dir,
        compute_pca=args.compute_pca,
        device=args.device,
        verbose=args.verbose,
        unique_name=args.unique_name,
        crop_slice=20
    )


if __name__ == '__main__':

    # with open("/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256.pkl", "rb") as f_rgb:
    #     dm = pickle.load(f_rgb)

    # x_min,x_max = dm.norm_stats['params'].values()
    # season_idx = dm.test_ds.input.season_idx

    # natl_test_data = xr.open_dataarray("/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc").isel(time=season_idx)
    # max_depth = 2000
    # # Drop all lat coordinates presenting a nan for depths (z) inferior to 2000
    # # Select only data for depths < 2000
    # sub_da = natl_test_data.sel(z=natl_test_data.z.where(natl_test_data.z < max_depth, drop=True))
    # # For each lat, check if there is any nan across time, z, and lon
    # lat_nan = sub_da.isnull().any(dim=["time", "z", "lon"])
    # # Get valid latitudes (i.e. where there is no nan)
    # valid_lats = lat_nan.where(lat_nan == False, drop=True).coords["lat"].values
    # # Select only the valid latitudes and drop all z coordinates superior to 2000.
    # natl_test_data = natl_test_data.sel(lat=valid_lats, z=natl_test_data.z.where(natl_test_data.z < max_depth, drop=True)).astype(getattr(np, dm.dtype_str))


    main()

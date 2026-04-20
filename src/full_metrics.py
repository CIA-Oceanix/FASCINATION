import sys
import os

running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)
sys.path.insert(0, "/Odyssey/private/o23gauvr/code/FASCINATION")
#from FASCINATION.src.autoencoder_datamodule_good_split import AEDatamodule

import numpy as np
import xarray as xr
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from MLIC.MLIC.models import MLICPlusPlus
from MLIC.MLIC.utils.utils import Config
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import convolve
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.stats import wasserstein_distance

from joblib import Parallel, delayed
from dtaidistance import dtw

from scipy.ndimage import gaussian_filter1d



from FASCINATION.src.utils import unorm_ssp_arr_3D

from sklearn.decomposition import PCA

import pandas as pd


from scipy.stats import pearsonr







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



def compute_total_bits(out_net) -> float:
    return sum(torch.log(likelihoods).sum() / (-math.log(2)) for likelihoods in out_net['likelihoods'].values()).item()




def compute_bpe(out_net, num_pixels) -> float:
    return sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    ).item()


def get_rmse_df_deciles(rmse_da: xr.DataArray, plot_path: Path, metric_type: str = "SSP") -> pd.DataFrame:

    # Flatten rmse_z and get valid (non-NaN) values with indices
    rmse_flat = rmse_da.values.ravel()
    valid_mask = np.isfinite(rmse_flat)
    valid_values = rmse_flat[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Compute 10 decile thresholds: 0%, 11.1%, 22.2%, ..., 100%
    percentiles = np.linspace(0, 100, 11)
    thresholds = np.percentile(valid_values, percentiles)

    # For each threshold, find the closest actual data point
    selected_flat_indices = []
    for thresh in thresholds:
        idx_in_valid = np.argmin(np.abs(valid_values - thresh))
        selected_flat_indices.append(valid_indices[idx_in_valid])

    # Convert flat indices back to multi-dimensional coordinates
    shape = rmse_da.shape
    dim_names = rmse_da.dims
    selected_points = []

    for i, flat_idx in enumerate(selected_flat_indices):
        multi_idx = np.unravel_index(flat_idx, shape)
        point = {dim: int(idx) for dim, idx in zip(dim_names, multi_idx)}  # integer indices
        point['rmse_value'] = rmse_flat[flat_idx]
        point['percentile'] = percentiles[i]
        selected_points.append(point)

    # Display results
    df_deciles = pd.DataFrame(selected_points)

    fig = plt.figure(figsize=(10, 6))
    plt.bar(df_deciles['percentile'], df_deciles['rmse_value'], width=3)
    plt.title(f'RMSE at Selected Percentiles ({metric_type})')
    plt.xlabel('Percentile')
    plt.ylabel('RMSE')
    plt.grid()
    plt.savefig(plot_path / metric_type.lower() / "rmse_deciles_bar_plot.png")
    plt.show()

    return df_deciles


def plot_rmse_per_depth(rmse_da: xr.DataArray, depth_array: np.ndarray, plot_path: Path, metric_type: str = "SSP"): 

    plt.figure(figsize=(8, 10))
    rmse_depth = rmse_da.mean(dim=["time", "lat", "lon"]).values
    std_depth = rmse_da.std(dim=["time", "lat", "lon"]).values
    plt.plot(rmse_depth, depth_array)
    plt.fill_betweenx(depth_array,
                        rmse_depth - std_depth,
                        rmse_depth + std_depth,
                        alpha=0.3)
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE")
    plt.ylabel("Depth (m)")
    plt.title(f"Depth-dependent RMSE ({metric_type})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path / metric_type.lower() / "rmse_per_depth.png", dpi=150, bbox_inches='tight')
    plt.show()






def plot_rmse_std(rmse_da: xr.DataArray, ssp_truth_std: xr.DataArray, plot_path: Path, metric_type: str = "SSP"):
    """Plot RMSE vs Truth Std Dev with correlation coefficient"""
    from scipy.stats import pearsonr
    
    # Flatten data and remove NaNs
    x = ssp_truth_std.values.ravel()
    y = rmse_da.values.ravel()
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    # Calculate correlation
    corr_coef, p_value = pearsonr(x, y)
    r_squared = corr_coef ** 2
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, s=20)
    
    # Add correlation info to plot
    textstr = f'Pearson r = {corr_coef:.4f}\nR² = {r_squared:.4f}\np-value = {p_value:.2e}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.title(f'RMSE vs {metric_type} Truth Standard Deviation')
    plt.xlabel(f'{metric_type} Standard Deviation')
    plt.ylabel('RMSE')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path / metric_type.lower() / "rmse_vs_std_scatter.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Correlation: r = {corr_coef:.4f}, R² = {r_squared:.4f}, p-value = {p_value:.2e}")


def get_dtw_arr(ssp_truth: np.ndarray, ssp_ae: np.ndarray) -> np.ndarray:
    flatten_truth = ssp_truth.transpose(0,2,3,1).reshape(-1, ssp_truth.shape[1])
    flatten_ae = ssp_ae.transpose(0,2,3,1).reshape(-1, ssp_ae.shape[1])
    dtw_arr = np.zeros(flatten_truth.shape[0])


    dtw_arr = Parallel(n_jobs=-1)(
        delayed(dtw.distance)(flatten_ae[prof], flatten_truth[prof]) 
        for prof in tqdm(range(flatten_truth.shape[0]))
    )
    dtw_arr = np.array(dtw_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])

    return dtw_arr


def get_wd_arr(power_truth: np.ndarray, power_ae: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    flatten_truth = power_truth.transpose(0,2,3,1).reshape(-1, power_truth.shape[1])
    flatten_ae = power_ae.transpose(0,2,3,1).reshape(-1, power_ae.shape[1])
    wd_arr = np.zeros(flatten_truth.shape[0])


    wd_arr = Parallel(n_jobs=-1)(
        delayed(wasserstein_distance)(freqs, freqs, flatten_ae[prof], flatten_truth[prof]) 
        for prof in tqdm(range(flatten_truth.shape[0]))
    )
    wd_arr = np.array(wd_arr).reshape(power_truth.shape[0], power_truth.shape[2], power_truth.shape[3])

    return wd_arr



def compute_power_spectrum(da, dim="z", detrend=True, window=True):
    """
    Compute power spectrum along a given dimension (default: z)
    """
    axis = da.get_axis_num(dim)
    n = da.sizes[dim]

    z_uniform = np.linspace(float(da.z.min()), float(da.z.max()), n)
    da = da.interp(z=z_uniform)

    coords = da[dim].values
    d = float(coords[1] - coords[0])

    data = da.values

    if detrend:
        data = data - np.mean(data, axis=axis, keepdims=True)

    # Apply windowing with correction factor
    if window:
        win = np.hanning(n)
        # Window energy correction for proper power scaling
        window_correction = n / np.sum(win**2)
        
        shape = [1] * data.ndim
        shape[axis] = n
        win = win.reshape(shape)
        data = data * win
    else:
        window_correction = 1.0

    fft_vals = np.fft.rfft(data, axis=axis)
    
    # Power spectrum with window correction
    power = (np.abs(fft_vals) ** 2) / (n * d) * window_correction
    # # Normalize to relative power (optional - remove if you want absolute power)
    # power = power / np.sum(power, axis=axis, keepdims=True)

    freqs = np.fft.rfftfreq(n, d=d)

    dims = list(da.dims)
    dims[axis] = "freq"

    coords = dict(da.coords)
    coords["freq"] = freqs
    coords.pop(dim)

    power_da = xr.DataArray(power, dims=dims, coords=coords, name="power_spectrum")

    return power_da, freqs


def plot_fft_analysis(power_da_truth, power_da_ae, freqs, plot_path: Path, metric_type: str = "SSP"):
    mean_power_truth = power_da_truth.mean(dim=("time", "lat", "lon"))
    mean_power_ae = power_da_ae.mean(dim=("time", "lat", "lon"))
    ratio = mean_power_ae / (mean_power_truth + 1e-12)

    eps = 1e-12
    log_truth = np.log(mean_power_truth + eps)
    log_rec   = np.log(mean_power_ae + eps)
    lsd = np.sqrt(np.mean((log_truth - log_rec)**2, axis=-1))

    peak_truth = np.argmax(mean_power_truth.data, axis=-1)
    peak_rec   = np.argmax(mean_power_ae.data, axis=-1)

    wd = wasserstein_distance(freqs, freqs,
                              mean_power_truth.data,
                              mean_power_ae.data)

    textstr = f'LSD = {lsd:.4f}\nPeak Freq Truth = {freqs[peak_truth]:.4f}\nPeak Freq AE = {freqs[peak_rec]:.4f}\nWasserstein Dist = {wd:.4f}'


    plt.figure(figsize=(10, 6))
    plt.plot(mean_power_truth.freq, mean_power_truth, label="Truth")
    plt.plot(mean_power_ae.freq, mean_power_ae, label="AE")
    plt.xlabel("Frequency (1/m)")
    plt.ylabel("Mean Power")
    plt.title("Mean Power Spectrum Comparison")
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_path / metric_type.lower() / "mean_power_spectrum_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    

    plt.figure(figsize=(10, 6))
    plt.plot(mean_power_truth.freq, ratio, label="AE/Truth Power Ratio")
    plt.xlabel("Frequency (1/m)")
    plt.ylabel("Power Ratio")  
    plt.title(f"Mean Power Spectrum Ratio (AE/Truth) - {metric_type}")
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_path / metric_type.lower() / "mean_power_spectrum_ratio.png", dpi=150, bbox_inches='tight')



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

    return np.prod(mssim ** weights)


def compute_ssim(ssp_truth: np.ndarray, ssp_ae: np.ndarray) -> np.ndarray:
    flatten_truth = ssp_truth.transpose(0,2,3,1).reshape(-1, ssp_truth.shape[1])
    flatten_ae = ssp_ae.transpose(0,2,3,1).reshape(-1, ssp_ae.shape[1])
    ssim_arr = []
    ssim_arr = Parallel(n_jobs=-1)(
        delayed(ssim_1d)(flatten_truth[prof], flatten_ae[prof]) 
        for prof in tqdm(range(flatten_truth.shape[0]))
    )

    ssim_arr = np.array(ssim_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    return ssim_arr


def compute_ms_ssim(ssp_truth: np.ndarray, ssp_ae: np.ndarray) -> np.ndarray:
    flatten_truth = ssp_truth.transpose(0,2,3,1).reshape(-1, ssp_truth.shape[1])
    flatten_ae = ssp_ae.transpose(0,2,3,1).reshape(-1, ssp_ae.shape[1])
    mssim_arr = []
    mssim_arr = Parallel(n_jobs=-1)(
    delayed(ms_ssim_1d)(flatten_truth[prof], flatten_ae[prof]) 
    for prof in tqdm(range(flatten_truth.shape[0])))
    mssim_arr = np.array(mssim_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    return mssim_arr

def plot_pca_analysis(pca, truth_pca, ae_pca, depth_array, plot_path: Path, metric_type: str = "SSP"):
    """Plot comprehensive PCA reconstruction analysis"""
    
    # Compute metrics per component
    rmse_per_mode = np.sqrt(((ae_pca - truth_pca) ** 2).mean(axis=0))
    explained_var = pca.explained_variance_
    explained_var_ratio = pca.explained_variance_ratio_
    std_per_mode = (ae_pca - truth_pca).std(axis=0)
    
    # Calculate relative error and R² per component
    relative_error = rmse_per_mode / np.sqrt(explained_var)
    r2_per_mode = 1 - (np.mean((ae_pca - truth_pca) ** 2, axis=0) / np.var(truth_pca, axis=0))
    
    # Pearson correlation per component
    pears_pca = np.array([pearsonr(truth_pca[:, i], ae_pca[:, i])[0] 
                          for i in range(truth_pca.shape[1])])
    
    # Create main comparison (6 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Explained variance cumulative sum
    cumsum_var = np.cumsum(explained_var_ratio)
    axes[0, 0].plot(np.arange(1, len(cumsum_var) + 1), cumsum_var, marker='o', linewidth=2, markersize=8)
    axes[0, 0].fill_between(np.arange(1, len(cumsum_var) + 1), cumsum_var, alpha=0.3)
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[0, 0].set_xlabel('PC Component')
    axes[0, 0].set_ylabel('Cumulative Explained Variance Ratio')
    axes[0, 0].set_title('Cumulative Explained Variance')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # 2. RMSE per mode
    axes[0, 1].errorbar(np.arange(1, len(rmse_per_mode) + 1), rmse_per_mode, 
                         yerr=std_per_mode, marker='o', capsize=4, linewidth=1.5)
    axes[0, 1].set_xlabel('PC Component')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Reconstruction Error per Component')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Relative error (normalized by variance)
    axes[0, 2].plot(np.arange(1, len(relative_error) + 1), relative_error, 
                    marker='o', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('PC Component')
    axes[0, 2].set_ylabel('Relative Error (RMSE / σ)')
    axes[0, 2].set_title('Normalized Reconstruction Error')
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Error vs explained variance
    axes[1, 0].scatter(explained_var, rmse_per_mode, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    axes[1, 0].set_xlabel('Explained Variance')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Error vs Importance')
    for i, txt in enumerate(np.arange(1, len(rmse_per_mode) + 1)):
        axes[1, 0].annotate(txt, (explained_var[i], rmse_per_mode[i]), fontsize=9)
    axes[1, 0].grid(alpha=0.3)
    
    # 5. R² per component
    axes[1, 1].bar(np.arange(1, len(r2_per_mode) + 1), r2_per_mode, alpha=0.7, color='steelblue')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1, 1].set_xlabel('PC Component')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Reconstruction Quality (R²) per Component')
    axes[1, 1].set_ylim([-0.1, 1.05])
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Pearson correlation per component
    axes[1, 2].bar(np.arange(1, len(pears_pca) + 1), pears_pca, alpha=0.7, color='green')
    axes[1, 2].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1, 2].axhline(y=1.0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 2].set_xlabel('PC Component')
    axes[1, 2].set_ylabel('Pearson Correlation')
    axes[1, 2].set_title('Correlation (Truth vs AE) per Component')
    axes[1, 2].set_ylim([0.7, 1.02])
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path / metric_type.lower() / "pca_reconstruction_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Separate figure: PCA components along depth
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(rmse_per_mode)))
    
    for i in range(len(rmse_per_mode)):
        ax.plot(pca.components_[i, :], depth_array, label=f'PC {i+1} ({explained_var_ratio[i]*100:.1f}%)', 
                color=colors[i], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Component Value', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'All PCA Components along Depth ({metric_type})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_path / metric_type.lower() / "pca_components_all_modes.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n=== PCA Analysis Summary ===")
    print(f"Cumulative explained variance (95%): {cumsum_var[np.where(cumsum_var >= 0.95)[0][0]]} at component {np.where(cumsum_var >= 0.95)[0][0] + 1}")
    print("\nPer-component metrics:")
    for i in range(len(rmse_per_mode)):
        print(f"PC {i+1}: Var={explained_var_ratio[i]*100:.2f}%, RMSE={rmse_per_mode[i]:.4f}, R²={r2_per_mode[i]:.4f}, Pearson r={pears_pca[i]:.4f}")


def cdist_extremum(profile_idx, flat_truth_ext, flat_ae_ext, depth):

    truth_idx = np.where(flat_truth_ext[profile_idx])[0]
    ae_idx = np.where(flat_ae_ext[profile_idx])[0]

    if len(truth_idx) == 0 or len(ae_idx) == 0:
        return np.nan

    D = torch.cdist(
        depth[truth_idx][:,None],
        depth[ae_idx][:,None]
    )

    return D.min(axis=1).mean()
    
def get_extremum_position_error(ssp_truth, ssp_ae, depth: np.ndarray, profile_idx: int) -> float:

    # Compute extrema masks
    ext_truth = get_min_max_idx(ssp_truth.values, axs=1, pad=True).astype(bool)
    ext_ae = get_min_max_idx(ssp_ae.values, axs=1, pad=True).astype(bool)

    # Flatten exactly like your DTW code
    flat_truth_ext = ext_truth.transpose(0,2,3,1).reshape(-1, ext_truth.shape[1])
    flat_ae_ext = ext_ae.transpose(0,2,3,1).reshape(-1, ext_ae.shape[1])


    # Parallel computation
    error_arr = Parallel(n_jobs=-1)(
        delayed(cdist_extremum)(prof, flat_truth_ext, flat_ae_ext, depth)
        for prof in tqdm(range(flat_truth_ext.shape[0]))
    )

    error_arr = np.array(error_arr).reshape(ssp_truth.shape[0], ssp_truth.shape[2], ssp_truth.shape[3])
    return error_arr


def plot_cluster_metric(shape_cluster_coords: dict, metric_da: xr.DataArray, title: str, plot_path: Path, ylabel: str):
    """
    Plot metric values per cluster defined by shape_cluster_coords.
    
    Args:
        shape_cluster_coords: dict with cluster_id -> list of (lat, lon) tuples
        metric_da: DataArray with shape (time, lat, lon)
        title: Plot title
        plot_path: Path to save plot
        ylabel: Label for y-axis
    """
    plot_path.mkdir(parents=True, exist_ok=True)
    
    cluster_means = []
    cluster_stds = []
    cluster_ids = []
    
    for cluster_id, coords in shape_cluster_coords.items():
        # Extract values at cluster coordinates
        values = []
        for lat, lon in coords:
            try:
                val = metric_da.isel(lat=lat, lon=lon).values
                if np.isfinite(val).any():  # Only include finite values
                    if isinstance(val, np.ndarray):
                        values.extend(val[np.isfinite(val)])
                    else:
                        values.append(val)
            except (IndexError, KeyError):
                continue
        
        if values:
            cluster_means.append(np.nanmean(values))
            cluster_stds.append(np.nanstd(values))
            cluster_ids.append(cluster_id)

    cluster_df = pd.DataFrame({
        'cluster_id': cluster_ids,
        'mean': cluster_means,
        'std': cluster_stds
    })

    cluster_df = cluster_df.sort_values('cluster_id')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(cluster_df['cluster_id'], cluster_df['mean'], yerr=cluster_df['std'], capsize=5, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.set_xticks(cluster_df['cluster_id'])
    ax.set_xticklabels([f'Cluster {cid}' for cid in cluster_df['cluster_id']], rotation=45)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(cluster_df['mean'], cluster_df['std'])):
        ax.text(i+1, mean + std + 0.02*(max(cluster_df['mean'])-min(cluster_df['mean'])), 
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_name = title.lower().replace(' ', '_').replace('/', '_') + '.png'
    plt.savefig(plot_path / plot_name, dpi=150, bbox_inches='tight')
    plt.show()
    
    return pd.DataFrame({
        'cluster_id': cluster_ids,
        'mean': cluster_means,
        'std': cluster_stds
    })



def plot_cluster_metric(shape_cluster_coords, da, title: str, plot_path: Path, ylabel: str):

    mean_metric = da.mean().item()
    # Calculate mean metric per cluster and create scatter plot
    cluster_ids = []
    mean_metric_list = []

    for cluster_id, coords_list in tqdm(shape_cluster_coords.items()):
        metric_values = []
        for t, lat, lon in coords_list:
            # Extract metric value at this coordinate
            da = da.isel(time=t, lat=lat, lon=lon).values
            # Only include valid (non-NaN) values
            if np.isfinite(da):
                metric_values.append(da)

        if len(metric_values) > 0:
            mean_metric = np.mean(metric_values)
            cluster_ids.append(cluster_id)
            mean_metric_list.append(mean_metric)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(cluster_ids, mean_metric_list, s=100, alpha=0.6, edgecolors='black')
    plt.xlabel("Cluster ID")
    plt.ylabel(ylabel)
    plt.title(f"{title}\n(Mean across all clusters = {np.mean(mean_metric_list):.4f})")
    plt.grid(alpha=0.3)
    plt.xticks(cluster_ids)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlic_ckpt=True
    ckpt_file = Path("/Odyssey/private/o23gauvr/code/MLIC/experiments/test_enatl_natl_mse/fixed_weight_loss_64_96_1.0_CR_10000.0_enatl_natl__mean_std/20260402_154931/checkpoints/best_checkpoint_bpp_loss.pth.tar")
    xp_name = "mlic_basic_loss"
    plot_path = Path("/Odyssey/private/o23gauvr/code/FASCINATION/imgs") / xp_name
    plot_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (plot_path / "ssp").mkdir(parents=True, exist_ok=True)
    (plot_path / "gradient").mkdir(parents=True, exist_ok=True)

    with open('/Odyssey/private/o23gauvr/code/FASCINATION/pickle/shape_cluster_coords.pkl', 'rb') as f:
        shape_cluster_coords = pickle.load(f)


    ## LOAD DATAMODULE ##
    dm_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split.pkl"  #"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_157_128_192_good_split.pkl" #"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_157_196_256_good_split.pkl"
    with open(dm_path, 'rb') as f:
        dm = pickle.load(f)


    ssp_truth_da = dm.test_ds.input
    depth_array = ssp_truth_da.z.values
    dm_test_norm = ssp_truth_da.attrs['norm_stats']
    season_idx = dm.test_ds.input.season_idx
    sst_test_norm  = dm.test_ds.input.attrs['sst'].data

    ssp_truth_da  = unorm_ssp_arr_3D(ssp_truth_da, dm_test_norm)
    ssp_truth = ssp_truth_da.values.astype(np.float32)

    mixing_layer_idx = 60

    ## LOAD CHECKPOINTS ##

    
    if mlic_ckpt:
        cfg = parse_experiment_config(ckpt_file)
        N = cfg.get('N', 192)
        M = cfg.get('M', 320)
        ssp_config = Config({
            'N': N, 'M': M, 'slice_num': cfg.get('slice_num', 10),
            'context_window': cfg.get('context_window', 5), 'act': cfg.get('act', nn.GELU),
            'in_channels': cfg.get('in_channels', 157), 'out_channels': cfg.get('in_channels', 157), 
            "add_seasons": cfg.get("add_seasons", {"use": False, "mode": None}),
            "add_sst": cfg.get("add_sst", False)
        })

    

        net = MLICPlusPlus(config=ssp_config).eval().to(device)
        ck = torch.load(ckpt_file, map_location=device)

        net.load_state_dict(ck['state_dict'])


        test_norm = ck.get('train_norm_stats', None)
        
        #test_norm_method =   #test_norm['method']
        test_norm['method'] = ck.get('train_norm_stats', None)['method']
        #test_norm_method = "mean_std" #"mean_std_along_depth" #"mean_std"

        
        if test_norm['method'] == "min_max":
            x_min = test_norm["params"]["x_min"].astype(ssp_truth.dtype)
            x_max = test_norm["params"]["x_max"].astype(ssp_truth.dtype)    
            ssp_truth_tens = (ssp_truth - x_min) / (x_max - x_min)
        elif test_norm['method'] == "mean_std":
            mean = test_norm['params']["mean"].astype(ssp_truth.dtype)
            std = test_norm['params']["std"].astype(ssp_truth.dtype)
            ssp_truth_tens = (ssp_truth - mean) / std
        elif test_norm['method'] == "mean_std_along_depth":
            mean = test_norm['params']["mean_along_depth"].astype(ssp_truth.dtype)
            std = test_norm['params']["std_along_depth"].astype(ssp_truth.dtype)
            ssp_truth_tens = (ssp_truth - mean) / std


                
        ssp_truth_tens = torch.tensor(ssp_truth_tens).to(device=device, dtype=getattr(torch,dm.dtype_str))
        with torch.no_grad():
            rv_batch = net(ssp_truth_tens, season_idx, sst_test_norm)

        ssp_ae = rv_batch['x_hat'].detach().cpu().numpy()

        ssp_ae = unorm_ssp_arr_3D(ssp_ae, test_norm)
        #ssp_truth = unorm_ssp_arr_3D(ssp_truth, test_norm)

        bits = compute_total_bits(rv_batch)
        total_original_bits = rv_batch['x_hat'].numel() * rv_batch['x_hat'].element_size() * 8

        cr = total_original_bits / bits

        ssp_ae_da = ssp_truth_da.copy(data=ssp_ae)


    ## POST PROCESSING ##
    crop_idx = slice(20,-20)

    ssp_truth_da = ssp_truth_da.isel(lat=crop_idx,lon=crop_idx)
    ssp_ae_da = ssp_ae_da.isel(lat=crop_idx,lon=crop_idx)

    b, a = butter(N=2, Wn=0.1, btype='low', analog=False)
    ssp_ae_da[:] = filtfilt(b, a, ssp_ae_da.data, axis=1).astype(ssp_ae_da.dtype)
    
    ssp_truth = ssp_truth_da.values.astype(np.float32)
    ssp_ae = ssp_ae_da.values.astype(np.float32)

    ## COMPUTE METRICS ##


    ### RMSE ###
    rmse_da = np.sqrt(((ssp_ae_da - ssp_truth_da) ** 2).mean(dim="z", skipna=True))
    ssp_truth_std = ssp_truth_da.std(dim="z", skipna=True)

    
    plot_rmse_per_depth(rmse_da, depth_array, plot_path=plot_path, metric_type="SSP")
    plot_rmse_std(rmse_da, ssp_truth_std, plot_path=plot_path, metric_type="SSP")

    rmse_df_deciles = get_rmse_df_deciles(rmse_da, plot_path=plot_path, metric_type="SSP")

    plot_cluster_metric(shape_cluster_coords, rmse_da, title="Mean RMSE per Shape-based cluster", plot_path=plot_path / "ssp", ylabel="RMSE (m/s)")
    plot_cluster_metric(shape_cluster_coords, rmse_da.isel(z=slice(0, mixing_layer_idx)), title=f"Mean RMSE per Shape-based cluster in mixing layer above {depth_array[mixing_layer_idx]} m", plot_path=plot_path / "ssp", ylabel="RMSE (m/s)")



    ### ECS ###
    max_ssp_truth_idx = np.nanargmax(ssp_truth, axis=1)
    max_ssp_ae_idx = np.nanargmax(ssp_ae, axis=1)
    ecs = np.abs(depth_array[max_ssp_truth_idx] - depth_array[max_ssp_ae_idx])
    ecs_da = xr.DataArray(ecs, coords=rmse_da.coords, dims=rmse_da.dims)

    plot_cluster_metric(shape_cluster_coords, ecs_da, title="Mean ECS per Shape-based cluster", plot_path=plot_path / "ssp", ylabel="ECS (m)")

    ### EXTREMUM POSITION CDIST ###
    extremum_position_error_arr = get_extremum_position_error(ssp_truth, ssp_ae, depth_array, profile_idx=0)
    extremum_position_error_da = xr.DataArray(extremum_position_error_arr, coords=rmse_da.coords, dims=rmse_da.dims)

    plot_cluster_metric(shape_cluster_coords, extremum_position_error_da, title="Mean Extremum Position CDist per Shape-based cluster", plot_path=plot_path / "ssp", ylabel="Extremum Position Error (m)")



    ### F1 SCORE ##
    min_max_idx_truth = get_min_max_idx(ssp_truth, axs=1, pad=False)
    min_max_idx_ae = get_min_max_idx(ssp_ae, axs=1, pad=False)
    F1_score = get_f1_score(min_max_idx_truth, min_max_idx_ae, axs=1, kernel_size=10)
    f1_da = xr.DataArray(F1_score, coords=rmse_da.coords, dims=rmse_da.dims)

    plot_cluster_metric(shape_cluster_coords, f1_da, title="Mean F1 Score per Shape-based cluster", plot_path=plot_path / "ssp", ylabel="F1 Score")
    plot_cluster_metric(shape_cluster_coords, f1_da.isel(z=slice(0, mixing_layer_idx)), title=f"Mean F1 Score per Shape-based cluster in mixing layer above {depth_array[mixing_layer_idx]} m", plot_path=plot_path / "ssp", ylabel="F1 Score")


    ### PEARSON CORRELATION ##
    pears = pearsonr(ssp_truth.transpose(1,0,2,3).reshape(ssp_truth.shape[1], -1), ssp_ae.transpose(1,0,2,3).reshape(ssp_ae.shape[1], -1))
    pearson_da = xr.DataArray(pears.statistic.reshape(rmse_da.shape), coords=rmse_da.coords, dims=rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, pearson_da, title="Mean Pearson Correlation per Shape-based cluster (SSP)", plot_path=plot_path / "ssp", ylabel="Pearson r")
    
    ### R² SCORE ##
    ss_res = ((ssp_ae_da - ssp_truth_da) ** 2).sum()
    ss_tot = ((ssp_truth_da - ssp_truth_da.mean()) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot)
    r2_da = xr.DataArray(
        np.full(rmse_da.shape, r2_score.values),
        coords=rmse_da.coords,
        dims=rmse_da.dims
    )
    plot_cluster_metric(shape_cluster_coords, r2_da, title="Mean R² Score per Shape-based cluster (SSP)", plot_path=plot_path / "ssp", ylabel="R²")

    ### DTW ###
    dtw_arr = get_dtw_arr(ssp_truth, ssp_ae)
    dtw_da = xr.DataArray(dtw_arr, coords=rmse_da.coords, dims=rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, dtw_da, title="Mean DTW per Shape-based cluster (SSP)", plot_path=plot_path / "ssp", ylabel="DTW Distance")


    ### FOURNIER ANALYSIS ###
    power_da_truth, freqs = compute_power_spectrum(ssp_truth_da, dim="z", detrend=True, window=True)
    power_da_ae, _ = compute_power_spectrum(ssp_ae_da, dim="z", detrend=True, window=True)

    plot_fft_analysis(power_da_truth, power_da_ae, freqs, plot_path=plot_path, metric_type="SSP")
    eps = 1e-12
    log_truth = np.log(power_da_truth + eps)
    log_rec   = np.log(power_da_ae + eps)
    lsd = np.sqrt(np.mean((log_truth - log_rec)**2, axis=1))

    peak_truth = np.argmax(power_da_truth.data, axis=1)
    peak_rec   = np.argmax(power_da_ae.data, axis=1)
    peak_freq_error = np.abs(freqs[peak_truth] - freqs[peak_rec])

    wd_arr = get_wd_arr(power_da_truth.data, power_da_ae.data, freqs)
    wd_da = xr.DataArray(wd_arr, coords=rmse_da.coords, dims=rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, wd_da, title="Mean Wasserstein Distance per Shape-based cluster (SSP)", plot_path=plot_path / "ssp", ylabel="Wasserstein Distance (1/m)")

    ### MS-SSIM ###
    ssim_1d_arr = compute_ssim(ssp_truth, ssp_ae)
    ssim_da = xr.DataArray(ssim_1d_arr, coords=rmse_da.coords, dims=rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, ssim_da, title="Mean 1D SSIM per Shape-based cluster (SSP)", plot_path=plot_path / "ssp", ylabel="SSIM")

    mssim_arr = compute_ms_ssim(ssp_truth, ssp_ae)
    mssim_da = xr.DataArray(mssim_arr, coords=rmse_da.coords, dims=rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, mssim_da, title="Mean MS-SSIM per Shape-based cluster (SSP)", plot_path=plot_path / "ssp", ylabel="MS-SSIM")


    ### PCA ANALYSIS ###
    n_components = 6
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    truth_pca = pca.fit_transform(ssp_truth.transpose(0,2,3,1).reshape(-1, ssp_truth.shape[1]))
    ae_pca = pca.transform(ssp_ae.transpose(0,2,3,1).reshape(-1, ssp_ae.shape[1]))
    plot_pca_analysis(pca, truth_pca, ae_pca, depth_array, plot_path=plot_path, metric_type="SSP")


#-----------------------------------------------

    # GRADIENT ANALYSIS #
    grad_truth_da = ssp_truth_da.differentiate("z")
    grad_ae_da = ssp_ae_da.differentiate("z")
    
    grad_truth_std = grad_truth_da.std(dim="z", skipna=True)
    
    ### GRADIENT RMSE ###
    grad_rmse_da = np.sqrt(((grad_ae_da - grad_truth_da) ** 2).mean(dim="z", skipna=True))
    
    plot_rmse_per_depth(grad_rmse_da, depth_array, plot_path=plot_path, metric_type="Gradient")
    plot_rmse_std(grad_rmse_da, grad_truth_std, plot_path=plot_path, metric_type="Gradient")
    
    grad_rmse_df_deciles = get_rmse_df_deciles(grad_rmse_da, plot_path=plot_path, metric_type="Gradient")
    plot_cluster_metric(shape_cluster_coords, grad_rmse_da, title="Mean Gradient RMSE per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="RMSE (1/s) per m")

    ### GRADIENT ECS ###
    max_grad_truth_idx = np.nanargmax(grad_truth_da.values, axis=1)
    max_grad_ae_idx = np.nanargmax(grad_ae_da.values, axis=1)
    grad_ecs = np.abs(depth_array[max_grad_truth_idx] - depth_array[max_grad_ae_idx])
    grad_ecs_da = xr.DataArray(grad_ecs, coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_ecs_da, title="Mean Gradient ECS per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="ECS (m)")

    ### GRADIENT EXTREMUM POSITION CDIST ##
    grad_extremum_position_error_arr = get_extremum_position_error(grad_truth_da.values, grad_ae_da.values, depth_array, profile_idx=0)
    grad_extremum_position_error_da = xr.DataArray(grad_extremum_position_error_arr, coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_extremum_position_error_da, title="Mean Gradient Extremum Position Error per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="Error (m)")
    
    ### GRADIENT F1 SCORE ##
    grad_truth = grad_truth_da.values.astype(np.float32)
    grad_ae = grad_ae_da.values.astype(np.float32)
    
    grad_min_max_idx_truth = get_min_max_idx(grad_truth, axs=1, pad=False)
    grad_min_max_idx_ae = get_min_max_idx(grad_ae, axs=1, pad=False)
    grad_F1_score = get_f1_score(grad_min_max_idx_truth, grad_min_max_idx_ae, axs=1, kernel_size=10)
    grad_f1_da = xr.DataArray(grad_F1_score, coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_f1_da, title="Mean Gradient F1 Score per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="F1 Score")
    
    ### GRADIENT PEARSON CORRELATION ##
    grad_pears = pearsonr(grad_truth.transpose(1,0,2,3).reshape(grad_truth.shape[1], -1), grad_ae.transpose(1,0,2,3).reshape(grad_ae.shape[1], -1))
    grad_pearson_da = xr.DataArray(grad_pears.statistic.reshape(grad_rmse_da.shape), coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_pearson_da, title="Mean Gradient Pearson Correlation per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="Pearson r")
    
    ### GRADIENT R² SCORE ###
    grad_ss_res = ((grad_ae_da - grad_truth_da) ** 2).sum()
    grad_ss_tot = ((grad_truth_da - grad_truth_da.mean()) ** 2).sum()
    grad_r2_score = 1 - (grad_ss_res / grad_ss_tot)
    grad_r2_da = xr.DataArray(
        np.full(grad_rmse_da.shape, grad_r2_score.values),
        coords=grad_rmse_da.coords,
        dims=grad_rmse_da.dims
    )
    plot_cluster_metric(shape_cluster_coords, grad_r2_da, title="Mean Gradient R² Score per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="R²")
    
    ### GRADIENT DTW ###
    grad_dtw_arr = get_dtw_arr(grad_truth, grad_ae)
    grad_dtw_da = xr.DataArray(grad_dtw_arr, coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_dtw_da, title="Mean Gradient DTW per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="DTW Distance")
    
    ### GRADIENT POWER SPECTRUM ###
    grad_power_da_truth, grad_freqs = compute_power_spectrum(grad_truth_da, dim="z", detrend=True, window=True)
    grad_power_da_ae, _ = compute_power_spectrum(grad_ae_da, dim="z", detrend=True, window=True)
    
    plot_fft_analysis(grad_power_da_truth, grad_power_da_ae, grad_freqs, plot_path=plot_path, metric_type="Gradient")
    
    grad_eps = 1e-12
    grad_log_truth = np.log(grad_power_da_truth + grad_eps)
    grad_log_rec = np.log(grad_power_da_ae + grad_eps)
    grad_lsd = np.sqrt(np.mean((grad_log_truth - grad_log_rec)**2, axis=1))
    
    grad_peak_truth = np.argmax(grad_power_da_truth.data, axis=1)
    grad_peak_rec = np.argmax(grad_power_da_ae.data, axis=1)
    grad_peak_freq_error = np.abs(grad_freqs[grad_peak_truth] - grad_freqs[grad_peak_rec])
    
    grad_wd_arr = get_wd_arr(grad_power_da_truth.data, grad_power_da_ae.data, grad_freqs)
    grad_wd_da = xr.DataArray(grad_wd_arr, coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_wd_da, title="Mean Gradient Wasserstein Distance per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="Wasserstein Distance (1/m)")
    
    ### GRADIENT MS-SSIM ###
    grad_ssim_1d_arr = compute_ssim(grad_truth, grad_ae)
    grad_ssim_da = xr.DataArray(grad_ssim_1d_arr, coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_ssim_da, title="Mean Gradient 1D SSIM per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="SSIM")
    
    grad_mssim_arr = compute_ms_ssim(grad_truth, grad_ae)
    grad_mssim_da = xr.DataArray(grad_mssim_arr, coords=grad_rmse_da.coords, dims=grad_rmse_da.dims)
    plot_cluster_metric(shape_cluster_coords, grad_mssim_da, title="Mean Gradient MS-SSIM per Shape-based cluster", plot_path=plot_path / "gradient", ylabel="MS-SSIM")
    
    ### GRADIENT PCA ANALYSIS ###
    grad_n_components = 6
    grad_pca = PCA(n_components=grad_n_components, svd_solver='randomized', random_state=42)
    grad_truth_pca = grad_pca.fit_transform(grad_truth.transpose(0,2,3,1).reshape(-1, grad_truth.shape[1]))
    grad_ae_pca = grad_pca.transform(grad_ae.transpose(0,2,3,1).reshape(-1, grad_ae.shape[1]))
    plot_pca_analysis(grad_pca, grad_truth_pca, grad_ae_pca, depth_array, plot_path=plot_path, metric_type="Gradient")


    # Combine all metrics into a single Dataset
    metrics_ds = xr.Dataset({
        # SSP Metrics
        'ssp_rmse': rmse_da,
        'ssp_ecs': ecs_da,
        'ssp_extremum_position_error': extremum_position_error_da,
        'ssp_f1_score': f1_da,
        'ssp_r2_score': r2_da,
        'ssp_pearson_correlation': pearson_da,
        'ssp_dtw': dtw_da,
        'ssp_wasserstein_distance': wd_da,
        'ssp_ssim_1d': ssim_da,
        'ssp_ms_ssim': mssim_da,
        'ssp_log_spectral_distance': lsd,
        'ssp_peak_freq_error': peak_freq_error,
        
        # Gradient Metrics
        'grad_rmse': grad_rmse_da,
        'grad_ecs': grad_ecs_da,
        'grad_extremum_position_error': grad_extremum_position_error_da,
        'grad_f1_score': grad_f1_da,
        'grad_r2_score': grad_r2_da,
        'grad_pearson_correlation': grad_pearson_da,
        'grad_dtw': grad_dtw_da,
        'grad_wasserstein_distance': grad_wd_da,
        'grad_ssim_1d': grad_ssim_da,
        'grad_ms_ssim': grad_mssim_da,
        'grad_log_spectral_distance': grad_lsd,
        'grad_peak_freq_error': grad_peak_freq_error,
    })

    # Optionally save to NetCDF
    output_file = Path("/Odyssey/private/o23gauvr/code/FASCINATION/pickle/") / f"metrics_dataset_{xp_name}.nc"
    metrics_ds.to_netcdf(output_file)
    print(f"✓ Saved metrics dataset to {output_file}")
    print(f"\nDataset summary:\n{metrics_ds}")

    # Print comprehensive metrics summary with units
    print("\n" + "="*100)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*100)
    
    print("\n### SSP METRICS (Sound Speed Profile - unit: m/s) ###")
    print(f"  RMSE:                         {rmse_da.mean().values:.6f} m/s")
    print(f"  R² Score:                     {r2_da.mean().values:.6f} (dimensionless)")
    print(f"  Extremum Chromatic Shift:     {ecs_da.mean().values:.6f} m")
    print(f"  Extremum Position Error:      {extremum_position_error_da.mean().values:.6f} m")
    print(f"  F1 Score (Extrema):           {f1_da.mean().values:.6f} (dimensionless)")
    print(f"  Pearson Correlation:          {pearson_da.mean().values:.6f} (dimensionless)")
    print(f"  Dynamic Time Warping:         {dtw_da.mean().values:.6f} (distance metric)")
    print(f"  Wasserstein Distance:         {wd_da.mean().values:.6f} (1/m)")
    print(f"  1D SSIM:                      {ssim_da.mean().values:.6f} (dimensionless)")
    print(f"  MS-SSIM:                      {mssim_da.mean().values:.6f} (dimensionless)")
    print(f"  Log Spectral Distance:        {lsd.mean().values:.6f} (dimensionless)")
    print(f"  Peak Frequency Error:         {peak_freq_error.mean().values:.6e} (1/m)")
    
    print("\n### GRADIENT METRICS (dSSP/dz - unit: 1/s) ###")
    print(f"  RMSE:                         {grad_rmse_da.mean().values:.6f} (1/s) per m")
    print(f"  R² Score:                     {grad_r2_da.mean().values:.6f} (dimensionless)")
    print(f"  Extremum Chromatic Shift:     {grad_ecs_da.mean().values:.6f} m")
    print(f"  Extremum Position Error:      {grad_extremum_position_error_da.mean().values:.6f} m")
    print(f"  F1 Score (Extrema):           {grad_f1_da.mean().values:.6f} (dimensionless)")
    print(f"  Pearson Correlation:          {grad_pearson_da.mean().values:.6f} (dimensionless)")
    print(f"  Dynamic Time Warping:         {grad_dtw_da.mean().values:.6f} (distance metric)")
    print(f"  Wasserstein Distance:         {grad_wd_da.mean().values:.6f} (1/m)")
    print(f"  1D SSIM:                      {grad_ssim_da.mean().values:.6f} (dimensionless)")
    print(f"  MS-SSIM:                      {grad_mssim_da.mean().values:.6f} (dimensionless)")
    print(f"  Log Spectral Distance:        {grad_lsd.mean().values:.6f} (dimensionless)")
    print(f"  Peak Frequency Error:         {grad_peak_freq_error.mean().values:.6e} (1/m)")
    
    print("\n### COMPRESSION METRICS ###")
    print(f"  Compression Ratio (CR):       {cr:.2f}")
    print(f"  Bits Per Element (BPE):       {bits / rv_batch['x_hat'].numel():.6f} bits/element")
    
    print("\n" + "="*100)

    print("Analysis complete!")
import sys
import os

running_path = "/Odyssey/private/o23gauvr/code/"
sys.path.insert(0, running_path)
os.chdir(running_path)

import hydra
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from FASCINATION.src.utils import *
from src.differentiable_fonc import Differentiable4dPCA
import torch.nn.functional as F
from scipy.ndimage import convolve
from scipy.interpolate import interp1d
from pytorch_msssim import ms_ssim
import torch

def get_min_max_idx(arr, axs=1, pad=True):
    grad = np.diff(arr, axis=axs)
    grad_sign = np.sign(grad)
    min_max = np.diff(grad_sign, axis=axs)
    min_max = np.abs(np.sign(min_max))
    if pad:
        min_max = np.pad(min_max, ((0, 0), (1, 1), (0, 0), (0, 0)), 'constant', constant_values=1)
    return min_max

def calculate_confusion_matrix_and_f1_score(min_max_idx_truth, min_max_idx_pca, axs=1, as_ratio=False):
    kernel_shape = [1] * min_max_idx_truth.ndim
    kernel_shape[axs] = 7
    kernel = np.ones(kernel_shape)
    truth_expanded = convolve(min_max_idx_truth, kernel, mode='constant', cval=0.0)
    pca_expanded = convolve(min_max_idx_pca, kernel, mode='constant', cval=0.0)

    true_positives = (truth_expanded > 0) & (min_max_idx_pca > 0)
    num_true_positives = np.sum(true_positives)
    false_positives = (truth_expanded == 0) & (min_max_idx_pca > 0)
    num_false_positives = np.sum(false_positives)
    true_negatives = (min_max_idx_truth == 0) & (min_max_idx_pca == 0)
    num_true_negatives = np.sum(true_negatives)
    false_negatives = (min_max_idx_truth > 0) & (pca_expanded == 0)
    num_false_negatives = np.sum(false_negatives)

    confusion_matrix = np.array([[num_true_negatives, num_false_positives],
                                 [num_false_negatives, num_true_positives]])

    if as_ratio:
        total = np.sum(confusion_matrix)
        confusion_matrix = confusion_matrix / total

    precision_score = num_true_positives / (num_true_positives + num_false_positives) if (num_true_positives + num_false_positives) > 0 else 0
    recall_score = num_true_positives / (num_true_positives + num_false_negatives) if (num_true_positives + num_false_negatives) > 0 else 0
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0

    return confusion_matrix, f1_score

def cubic_interpolate_along_axis(arr: np.ndarray, target_size: int, axis: int) -> np.ndarray:
    current_size = arr.shape[axis]
    x_old = np.linspace(0, 1, current_size)
    x_new = np.linspace(0, 1, target_size)
    arr_swapped = np.moveaxis(arr, axis, 0)
    reshaped = arr_swapped.reshape(current_size, -1)
    f = interp1d(x_old, reshaped, kind='cubic', axis=0, bounds_error=False, fill_value="extrapolate")
    interpolated = f(x_new)
    new_shape = (target_size,) + arr_swapped.shape[1:]
    interpolated = interpolated.reshape(new_shape)
    return np.moveaxis(interpolated, 0, axis)

def compute_psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return -10 * np.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

class NoConvAE(nn.Module):
    def __init__(self, n: int, pooling_dim: str = "spatial", pooling_mode: str = "Avg"):
        super().__init__()
        self.pooling_dim = pooling_dim
        self.upsample_mode = "trilinear"
        if pooling_dim == "all":
            pool_str = (2, 1, 1)
        elif pooling_dim == "spatial":
            pool_str = (1, 2, 2)
        elif pooling_dim is None:
            pool_str = 1
        pooling_dict = {"Avg": nn.AvgPool3d(kernel_size=1, stride=pool_str, padding=0),
                        "Max": nn.MaxPool3d(kernel_size=1, stride=pool_str, padding=0),
                        "None": nn.Identity()}
        upsample_dict = {"Avg": nn.Upsample(scale_factor=pool_str, mode=self.upsample_mode),
                         "Max": nn.Upsample(scale_factor=pool_str, mode=self.upsample_mode),
                         "None": nn.Identity()}
        pool_layer = pooling_dict[pooling_mode]
        upsample_layer = upsample_dict[pooling_mode]
        self.encoder = nn.Sequential(*[pool_layer for _ in range(n)])
        self.decoder = nn.Sequential(*[upsample_layer for _ in range(n - 1)])
        self.decoder.append(nn.Upsample(size=None, mode=self.upsample_mode))

    def forward(self, x):
        if self.pooling_dim == "all":
            x = x.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        x = x.unsqueeze(1)
        self.decoder[-1].size = x.shape[2:]
        self.bottleneck = self.encoder(x)
        self.output = self.decoder(self.bottleneck).squeeze(1)
        if self.pooling_dim == "all":
            self.output = self.output.squeeze(-1).squeeze(-1)
            self.output = self.output.transpose(0, 1)
        return self.output

if __name__ == "__main__":
    verbose = True

    xp = "autoencoder_V2"
    use_4D_dif_pca = False #or xp == "autoencoder_V2"
    pooling_dim = "spatial" if xp == "autoencoder_V2" else "all"
    min_components = 1
    n_layers = 4
    gpu = 0
    cfg_path = f"config/xp/{xp}.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.dtype = "float64"
    if torch.cuda.is_available() and gpu is not None:
        dev = f"cuda:{gpu}"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("Inititing datamodule; Generating train and test datasets")
    dm = hydra.utils.call(cfg.datamodule)
    train_ssp_arr, _, dm = loading_datamodule(dm)
    if dm.norm_stats["norm_location"] == "datamodule":
        train_ssp_arr = unorm_ssp_arr_3D(train_ssp_arr, dm) #, unorm_ssp_arr_3D(test_ssp_arr, dm)
    


    #x_min,x_max = dm_mlic.norm_stats['params'].values()
    season_idx = dm.test_da.season_idx

    natl_test_data = xr.open_dataarray("/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc").isel(time=season_idx)
    max_depth = 2000
    # Drop all lat coordinates presenting a nan for depths (z) inferior to 2000
    # Select only data for depths < 2000
    sub_da = natl_test_data.sel(z=natl_test_data.z.where(natl_test_data.z < max_depth, drop=True))
    # For each lat, check if there is any nan across time, z, and lon
    lat_nan = sub_da.isnull().any(dim=["time", "z", "lon"])
    # Get valid latitudes (i.e. where there is no nan)
    valid_lats = lat_nan.where(lat_nan == False, drop=True).coords["lat"].values
    # Select only the valid latitudes and drop all z coordinates superior to 2000.
    natl_test_data = natl_test_data.sel(lat=valid_lats, z=natl_test_data.z.where(natl_test_data.z < max_depth, drop=True)).astype(getattr(np, dm.dtype_str))

    test_ssp_arr = natl_test_data.data
    
    test_ssp_tens = torch.tensor(test_ssp_arr, dtype=getattr(torch, cfg.dtype), device=device)
    
    max_components = train_ssp_arr.shape[1] + 1
    if min_components == -1:
        min_components = max_components - 1
    depth_array = dm.depth_array
    ecs_truth_idx = np.argmax(test_ssp_arr, axis=1)
    ecs_truth = depth_array[ecs_truth_idx]
    model_metrics = {}
    for n_components in tqdm(range(min_components, max_components), unit="components", desc="Computing PCA components", disable=not(verbose)):
        pca = PCA(n_components=n_components, svd_solver='auto')
        if xp == "autoencoder_V2":
            train_data = train_ssp_arr.transpose(0, 2, 3, 1).reshape(-1, train_ssp_arr.shape[1])
        else:
            train_data = train_ssp_arr
        pca.fit(train_data)
        if xp == "autoencoder_V2":
            test_data = test_ssp_arr.transpose(0, 2, 3, 1).reshape(-1, test_ssp_arr.shape[1])
        else:
            test_data = test_ssp_arr
        if use_4D_dif_pca:
            dif_pca = Differentiable4dPCA(pca, batch_shape=test_ssp_tens.shape, device=test_ssp_tens.device, dtype=test_ssp_tens.dtype)
            pca_reduced_test_ssp_tens = dif_pca.transform(test_ssp_tens)
        else:
            reduced = pca.transform(test_data)
            if xp == "autoencoder_V2":
                pca_reduced_test_ssp_tens = torch.tensor(
                    reduced.reshape(test_ssp_arr.shape[0], test_ssp_arr.shape[2], test_ssp_arr.shape[3], n_components).transpose(0, 3, 1, 2),
                    dtype=test_ssp_tens.dtype, device=test_ssp_tens.device)
            else:
                pca_reduced_test_ssp_tens = torch.tensor(reduced, dtype=test_ssp_tens.dtype, device=test_ssp_tens.device)
        for n_layer in tqdm(range(n_layers), disable=not(verbose), unit="layers", desc="Computing AE layers"):
            model_ae = NoConvAE(n_layer, pooling_dim=pooling_dim, pooling_mode="Avg")
            pooled_upsampled_test_ssp_tens = model_ae(pca_reduced_test_ssp_tens)
            if xp == "autoencoder_V2":
                lat_lon_shape = model_ae.bottleneck.squeeze(1).shape[-2:]
            if use_4D_dif_pca:
                pca_unreduced_test_ssp_tens = dif_pca.inverse_transform(pooled_upsampled_test_ssp_tens)
                pca_unreduced_test_ssp_arr = pca_unreduced_test_ssp_tens.detach().cpu().numpy()
            else:
                unreduced = pca.inverse_transform(pooled_upsampled_test_ssp_tens.detach().cpu().numpy())
                if xp == "autoencoder_V2":
                    pca_unreduced_test_ssp_arr = unreduced.reshape(test_ssp_arr.shape[0], test_ssp_arr.shape[2], test_ssp_arr.shape[3], test_ssp_arr.shape[1]).transpose(0, 3, 1, 2)
                else:
                    pca_unreduced_test_ssp_arr = unreduced
            # RMSE
            ae_ssp_rmse = np.sqrt(np.mean((test_ssp_arr - pca_unreduced_test_ssp_arr) ** 2))
            # ECS
            ecs_pred_idx = np.argmax(pca_unreduced_test_ssp_arr, axis=1)
            ecs_pred = depth_array[ecs_pred_idx]
            ae_ecs_rmse = np.sqrt(np.mean((ecs_truth - ecs_pred) ** 2))
            # MAE
            mae = np.mean(np.abs(test_ssp_arr - pca_unreduced_test_ssp_arr))
            # mean_error_n_min_max
            min_max_idx_truth = get_min_max_idx(test_ssp_arr, pad=False)
            min_max_idx_pca = get_min_max_idx(pca_unreduced_test_ssp_arr, pad=False)
            mean_error_n_min_max = np.mean(np.abs(np.sum(min_max_idx_truth, axis=1) - np.sum(min_max_idx_pca, axis=1)))
            # confusion matrix & F1
            conf_matrix_counts, f1_score = calculate_confusion_matrix_and_f1_score(min_max_idx_truth, min_max_idx_pca, as_ratio=False)
            # Filtered_F1_score placeholder (set to np.nan unless logic provided)
            filtered_f1_score = np.nan
            # R2
            r2 = 1 - (np.sum((test_ssp_arr - pca_unreduced_test_ssp_arr) ** 2) / np.sum((test_ssp_arr - np.mean(test_ssp_arr)) ** 2))
            # PSNR
            psnr = compute_psnr(test_ssp_arr, pca_unreduced_test_ssp_arr)
            # MS-SSIM (interpolate to 161 along axis=2)
            try:
                arr1 = cubic_interpolate_along_axis(test_ssp_arr, 161, axis=2)
                arr2 = cubic_interpolate_along_axis(pca_unreduced_test_ssp_arr, 161, axis=2)
                arr1_t = torch.tensor(arr1, dtype=torch.float64)
                arr2_t = torch.tensor(arr2, dtype=torch.float64)
                msssim = compute_msssim(arr1_t, arr2_t)
            except Exception as e:
                print(f"MS-SSIM computation failed: {e}")
                msssim = np.nan



            bits_per_value = 64 if cfg.dtype == "float64" else 32
            original_shape = test_ssp_arr.shape
            original_bits = np.prod(original_shape) * bits_per_value

            if n_layer == 0:
                model = "PCA"
                pooled_shape =  (original_shape[0], n_components , original_shape[2], original_shape[3])
            else:
                factor = 2 ** n_layer
                model = f"PCA pooled by factor {factor}x{factor}"
                pooling_factor = 2 ** n_layer
                pooled_shape = (original_shape[0], n_components,
                                original_shape[2] // pooling_factor,
                                original_shape[3] // pooling_factor)

            
            compressed_bits = np.prod(pooled_shape) * bits_per_value

            cr = original_bits / compressed_bits

            # Initialize nested dicts if needed
            if model not in model_metrics:
                model_metrics[model] = {}
            if cr not in model_metrics[model]:
                model_metrics[model][cr] = {}

            # Save metrics
            model_metrics[model][cr]["RMSE"] = ae_ssp_rmse
            model_metrics[model][cr]["PSNR"] = psnr
            model_metrics[model][cr]["MS-SSIM"] = msssim
            model_metrics[model][cr]["ECS"] = ae_ecs_rmse
            model_metrics[model][cr]["MAE"] = mae
            model_metrics[model][cr]["mean_error_n_min_max"] = mean_error_n_min_max
            model_metrics[model][cr]["F1_score"] = f1_score
            model_metrics[model][cr]["Filtered_F1_score"] = filtered_f1_score
            model_metrics[model][cr]["R2_score"] = r2

    if use_4D_dif_pca:
        pca_name = "dif_pca"
    else:
        pca_name = "sklearn_pca"
    with open(f'pickle/model_metrics_pca.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)
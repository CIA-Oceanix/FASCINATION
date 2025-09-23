import os 
import sys

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
from tqdm import tqdm
from scipy.ndimage import convolve
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from pytorch_msssim import ms_ssim

import FASCINATION.src.utils as utils
from FASCINATION.src.utils import unorm_ssp_arr_3D

try:
    from MLIC.MLIC.models import MLICPlusPlus
    from MLIC.MLIC.utils.utils import Config
except Exception:
    MLICPlusPlus = None
    Config = dict


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


def compute_psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute PSNR using explicit data_range (max - min). If data are in [0,1], data_range=1.0.
    PSNR = 10 * log10( (data_range ** 2) / MSE )
    """
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')
    # protect against zero/negative ranges
    dr = float(data_range) if data_range is not None else 1.0
    if dr <= 0:
        dr = 1.0
    return 10 * math.log10((dr ** 2) / mse)


def compute_msssim(a: torch.Tensor, b: torch.Tensor) -> float:
    return ms_ssim(a, b, data_range=1.).item()


def compute_total_bits(out_net: Dict[str, torch.Tensor]) -> float:
    return sum(torch.log(likelihoods).sum() / (-math.log(2)) for likelihoods in out_net['likelihoods'].values()).item()


def parse_experiment_config(model_path: str) -> Dict[str, Any]:
    files = list(Path(model_path).rglob('train_*.log'))
    if not files:
        return {}
    try:
        with open(files[0], 'r') as f:
            lines = f.readlines()
        # previous code stored a cfg dict in line 2; attempt to eval safely
        for ln in lines:
            if 'INFO:' in ln and '{' in ln:
                try:
                    cfg = eval((ln.strip().split("INFO: ")[-1]).replace("<class 'torch.nn.modules.activation.","nn.").replace("'>",""))
                    if isinstance(cfg, dict):
                        return cfg
                except Exception:
                    continue
    except Exception:
        pass
    return {}


def find_first_level_dirs(base_dir: str) -> Dict[str, str]:
    result = {}
    for name in next(os.walk(base_dir))[1]:
        if name == 'mute':
            continue
        result[name] = os.path.join(base_dir, name)
    return result


def compute_and_save(
    dm_mlic_pkl: str,
    dm_cae_pkl: str,
    mlic_base_dir: str,
    other_ckpt_base: str,
    out_pickle_dir: str,
    device: str = None,
    psnr_range_mode: str = 'truth',
    psnr_range_value: float = None,
    verbose: bool = False,
    unique_name: bool = True,
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datamodules
    if verbose:
        print(f'Loading datamodule pickles: {dm_mlic_pkl} and {dm_cae_pkl}')
    with open(dm_mlic_pkl, 'rb') as f:
        dm_mlic = pickle.load(f)
    with open(dm_cae_pkl, 'rb') as f:
        dm_cae = pickle.load(f)

    depth_array = dm_mlic.depth_array
    test_ssp_arr = dm_mlic.test_da.data

    # containers
    bit_rates = {}
    outputs = {}
    rgb_models = set()

    # --- MLIC checkpoints ---
    mlic_ckpt_dict = find_first_level_dirs(mlic_base_dir)
    if verbose:
        print(f'Found {len(mlic_ckpt_dict)} first-level entries under {mlic_base_dir}')

    # Process SSP MLIC++ (in_channels==157)
    for model_name, model_path in tqdm(mlic_ckpt_dict.items(), desc='SSP MLIC++ models', disable=not verbose):
        ckpt_files = list(Path(model_path).rglob('checkpoint_best_loss.pth.tar'))
        if not ckpt_files:
            continue
        for ckpt_file in ckpt_files:
            if verbose:
                print(f'  Inspecting SSP models in {model_name} -> {model_path}, ckpt: {ckpt_file}')
            cfg = parse_experiment_config(model_path)
            if cfg.get('in_channels', None) != 157:
                continue
            N = cfg.get('N', 192)
            M = cfg.get('M', 320)
            ssp_config = Config({
                'N': N, 'M': M, 'slice_num': cfg.get('slice_num', 10),
                'context_window': cfg.get('context_window', 5), 'act': cfg.get('act', nn.GELU),
                'in_channels': 157, 'out_channels': 157
            })
            try:
                net = MLICPlusPlus(config=ssp_config).eval().to(device)
                ck = torch.load(ckpt_file, map_location=device)
                net.load_state_dict(ck['state_dict'])
                x = torch.tensor(test_ssp_arr.copy()).to(device)
                with torch.no_grad():
                    rv = net(x)
                bits = compute_total_bits(rv)
                numel = rv['x_hat'].numel()
                original_bits = numel * 8
                bpe = bits / numel
                cr = original_bits / bits
                if len(ckpt_files) > 1:
                    sub_name = "_" + (str(ckpt_file).split(f'{model_name}/')[-1]).split('/')[0]
                else:
                    sub_name = ""
                if unique_name:
                    dict_model_name = 'MLIC++ SSP' 
                else:
                    dict_model_name = (model_name.split("mlicpp_on_ssp_")[-1]).split("_min_max")[0] + sub_name
                bit_rates.setdefault(dict_model_name, {})[cr] = {
                    'bits_per_element': bpe, 'compression_rate': cr,
                    'total_compressed_bits': bits, 'total_original_bits': original_bits,
                    'model_config': f'N={N}, M={M}', 'original_model_name': model_name
                }
                reconstructed_arr = rv['x_hat'].detach().cpu().numpy()
                unorm_reconstructed_arr = unorm_ssp_arr_3D(reconstructed_arr, dm_mlic)
                outputs.setdefault(dict_model_name, {})[cr] = unorm_reconstructed_arr
            except Exception:
                continue

    # Process RGB MLIC++ (in_channels==3)
    for model_name, model_path in tqdm(mlic_ckpt_dict.items(), desc='RGB MLIC++ models', disable=not verbose):
        ckpt_files = list(Path(model_path).rglob('checkpoint_best_loss.pth.tar'))
        if not ckpt_files:
            continue
        for ckpt_file in ckpt_files:
            if verbose:
                print(f'  Inspecting RGB models in {model_name} -> {model_path}, ckpt: {ckpt_file}')
            cfg = parse_experiment_config(model_path)
            if cfg.get('in_channels', None) != 3:
                continue
            N = cfg.get('N', 192)
            M = cfg.get('M', 320)
            ssp_config = Config({'N': N, 'M': M, 'slice_num': cfg.get('slice_num', 10), 'context_window': cfg.get('context_window', 5), 'act': cfg.get('act', nn.GELU), 'in_channels': 3, 'out_channels': 3})
            try:
                net = MLICPlusPlus(config=ssp_config).eval().to(device)
                ck = torch.load(ckpt_file, map_location=device)
                net.load_state_dict(ck['state_dict'])
                reconstructed_arr = np.zeros(test_ssp_arr.shape)
                total_bits = 0
                total_elements = 0
                total_original_bits = 0
                n_imgs = len(depth_array) // 3
                for j in range(n_imgs):
                    start_idx = j * 3
                    end_idx = start_idx + 3
                    x = torch.tensor(test_ssp_arr[:, start_idx:end_idx, :, :].copy()).to(device)
                    with torch.no_grad():
                        rv = net(x)
                    bits = compute_total_bits(rv)
                    numel = rv['x_hat'].numel()
                    original_bits = numel * 8
                    total_bits += bits
                    total_elements += numel
                    total_original_bits += original_bits
                    reconstructed_arr[:, start_idx:end_idx] = rv['x_hat'].detach().cpu().numpy()
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
                unorm_reconstructed_arr = unorm_ssp_arr_3D(reconstructed_arr, dm_mlic)
                outputs.setdefault(dict_model_name, {})[cr] = unorm_reconstructed_arr
                # remember this model came from RGB MLIC processing
                rgb_models.add(dict_model_name)
            except Exception:
                continue

    # --- User-provided original MLICC checkpoint ---
    # This block loads the specific original MLIC checkpoint the user asked for,
    # runs inference in the same triple-channel loop as the RGB MLIC above,
    # and registers the outputs and bitrate info under the name 'original_mlicc'.
    orig_mlicc_ckpt = "/Odyssey/private/o23gauvr/code/MLIC/checkpoints/mlicpp_mse_q5_2960000.pth.tar"
    if MLICPlusPlus is not None and Path(orig_mlicc_ckpt).exists():
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

            reconstructed_arr = np.zeros(test_ssp_arr.shape)
            total_bits = 0
            total_elements = 0
            total_original_bits = 0
            n_imgs = len(depth_array) // 3
            for j in range(n_imgs):
                start_idx = j * 3
                end_idx = start_idx + 3
                x = torch.tensor(test_ssp_arr[:, start_idx:end_idx, :, :].copy()).to(device)
                with torch.no_grad():
                    rv = rgb_mlic_net(x)
                bits = compute_total_bits(rv)
                numel = rv['x_hat'].numel()
                original_bits = numel * 8
                total_bits += bits
                total_elements += numel
                total_original_bits += original_bits
                reconstructed_arr[:, start_idx:end_idx] = rv['x_hat'].detach().cpu().numpy()
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
            unorm_reconstructed_arr = unorm_ssp_arr_3D(reconstructed_arr, dm_mlic)
            outputs.setdefault(dict_model_name, {})[cr] = unorm_reconstructed_arr
            # mark as RGB-derived so cutted metrics are computed
            rgb_models.add(dict_model_name)
        except Exception as e:
            if verbose:
                print(f'  Could not load/run original_mlicc from {orig_mlicc_ckpt}: {e}')

    # --- Other models loaded from ckpt directories ---
    ckpt_dict = find_first_level_dirs(other_ckpt_base)
    test_ssp_truth_arr = dm_cae.test_da.data
    test_ssp_truth = torch.tensor(test_ssp_truth_arr, device=device, dtype=getattr(torch, dm_cae.dtype_str))
    if verbose:
        print(f'Found {len(ckpt_dict)} model directories under {other_ckpt_base}')
    for model_carac in tqdm(ckpt_dict.keys(), desc='Models', disable=not verbose):
        ckpt_list = list(Path(ckpt_dict[model_carac]).rglob('*.ckpt'))
        target_name = 'CAE' if unique_name else model_carac
        outputs.setdefault(target_name, {})
        bit_rates.setdefault(target_name, {})
        for ckpt_path in ckpt_list:
            if verbose:
                print(f'  Loading model checkpoint: {ckpt_path}')
            try:
                cfg = utils.get_cfg_from_ckpt_path(str(ckpt_path), pprint=False)
            except Exception:
                if verbose:
                    print(f'    Could not get config for {ckpt_path}, skipping')
                continue
            try:
                lit_model = utils.load_model(str(ckpt_path), dm_cae, test_ssp_truth, verbose=False)
            except Exception:
                if verbose:
                    print(f'    Could not load model {ckpt_path}, skipping')
                continue
            ssp_ae_test_arr = lit_model(test_ssp_truth).detach().cpu().numpy().astype(test_ssp_truth_arr.dtype)
            cr = lit_model.model_AE.cr
            bpe = lit_model.model_AE.bpe
            total_bits = lit_model.model_AE.total_bits
            if dm_cae.norm_stats.get('norm_location', '') == 'datamodule':
                ssp_ae_test_arr = utils.unorm_ssp_arr_3D(ssp_ae_test_arr, dm_cae)
            outputs[target_name][cr] = ssp_ae_test_arr
            bit_rates[target_name][cr] = {'bits_per_element': bpe, 'compression_rate': cr, 'total_compressed_bits': total_bits}

    # interpolate outputs to original_da resolution if needed (keeps previous logic)
    augmented_da = dm_mlic.test_da
    original_da = dm_cae.test_da
    test_ssp_truth = original_da.data
    test_ssp_truth = utils.unorm_ssp_arr_3D(test_ssp_truth, dm_cae)

    # compute truth range
    truth_min = float(np.nanmin(test_ssp_truth))
    truth_max = float(np.nanmax(test_ssp_truth))
    truth_range = truth_max - truth_min if truth_max > truth_min else 1.0

    if verbose:
        print('Interpolating outputs to original datamodule resolution (if needed)')
    for model in list(outputs.keys()):
        if 'CAE' not in model:
            for cr, reconstructed_arr in list(outputs[model].items()):
                if verbose:
                    print(f'  Interpolating model {model} @ CR {cr}')
                augmented_da[:] = reconstructed_arr
                reconstructed_arr = augmented_da.interp_like(original_da, method='nearest').data
                outputs[model][cr] = reconstructed_arr

    # if global mode, compute global range across outputs and truth
    global_range = truth_range
    if psnr_range_mode == 'global':
        gmin = truth_min
        gmax = truth_max
        for _, cr_dict in outputs.items():
            for arr in cr_dict.values():
                try:
                    a_min = float(np.nanmin(arr))
                    a_max = float(np.nanmax(arr))
                    if a_min < gmin:
                        gmin = a_min
                    if a_max > gmax:
                        gmax = a_max
                except Exception:
                    continue
        global_range = gmax - gmin if gmax > gmin else truth_range

    # Evaluate metrics
    model_metrics: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for model, cr_dict in tqdm(outputs.items(), desc='Evaluating models', disable=not verbose):
        model_metrics[model] = {}
        for cr, ae_ssp_test_arr in cr_dict.items():
            if verbose:
                print(f'Evaluating metrics for model {model} @ CR {cr}')
            ae_ssp_test = ae_ssp_test_arr
            max_ssp_truth_idx = np.nanargmax(test_ssp_truth, axis=1)
            ecs_truth = depth_array[max_ssp_truth_idx]
            # determine data_range for PSNR according to mode
            if psnr_range_mode == 'per':
                try:
                    a_min = float(np.nanmin(ae_ssp_test))
                    a_max = float(np.nanmax(ae_ssp_test))
                    dr = max(a_max, truth_max) - min(a_min, truth_min)
                except Exception:
                    dr = truth_range
            elif psnr_range_mode == 'global':
                dr = global_range
            elif psnr_range_mode == 'value':
                dr = float(psnr_range_value) if psnr_range_value is not None else truth_range
            else:  # 'truth'
                dr = truth_range

            psnr = compute_psnr(test_ssp_truth, ae_ssp_test, data_range=dr)
            msssim = compute_msssim(torch.tensor(cubic_interpolate_along_axis(test_ssp_truth, 161, axis=2), dtype=torch.float64), torch.tensor(cubic_interpolate_along_axis(ae_ssp_test, 161, axis=2), dtype=torch.float64))
            max_ssp_ae_idx = np.nanargmax(ae_ssp_test, axis=1)
            ecs_pred_ae = depth_array[max_ssp_ae_idx]
            ae_ssp_rmse = np.sqrt(np.mean((test_ssp_truth - ae_ssp_test) ** 2))
            ae_ecs_rmse = np.sqrt(np.mean((ecs_truth - ecs_pred_ae) ** 2))
            mae = np.mean(np.abs(test_ssp_truth - ae_ssp_test))
            min_max_idx_truth = get_min_max_idx(test_ssp_truth, pad=False)
            min_max_idx_ae = get_min_max_idx(ae_ssp_test, pad=False)
            mean_error_n_min_max = np.mean(np.abs(np.sum(min_max_idx_truth, axis=1) - np.sum(min_max_idx_ae, axis=1)))
            F1_score = get_f1_score(min_max_idx_truth, min_max_idx_ae)
            f1_score = np.mean(F1_score)
            r2 = 1 - (np.sum((test_ssp_truth - ae_ssp_test) ** 2) / np.sum((test_ssp_truth - np.mean(test_ssp_truth)) ** 2))
            b, a = butter(N=2, Wn=0.107, btype='low', analog=False)
            filtered_ae_test_arr = filtfilt(b, a, ae_ssp_test, axis=1)
            min_max_idx_filtered_ae = get_min_max_idx(filtered_ae_test_arr, pad=False)
            filtered_F1_score = np.mean(get_f1_score(min_max_idx_truth, min_max_idx_filtered_ae))
            model_metrics[model][cr] = {
                'RMSE': ae_ssp_rmse,
                'PSNR': psnr,
                'MS-SSIM': msssim,
                'ECS': ae_ecs_rmse,
                'MAE': mae,
                'mean_error_n_min_max': mean_error_n_min_max,
                'F1_score': f1_score,
                'Filtered_F1_score': filtered_F1_score,
                'R2_score': r2,
            }

            # Additional metrics for RGB MLIC models with last depth point removed
            if model in rgb_models:
                try:
                    truth_cut = test_ssp_truth[:, :-1, ...]
                    ae_cut = ae_ssp_test[:, :-1, ...]
                    depth_cut = depth_array[:-1]

                    # psnr range for cut
                    if psnr_range_mode == 'per':
                        try:
                            a_min = float(np.nanmin(ae_cut))
                            a_max = float(np.nanmax(ae_cut))
                            dr_cut = max(a_max, truth_max) - min(a_min, truth_min)
                        except Exception:
                            dr_cut = truth_range
                    elif psnr_range_mode == 'global':
                        dr_cut = global_range
                    elif psnr_range_mode == 'value':
                        dr_cut = float(psnr_range_value) if psnr_range_value is not None else truth_range
                    else:
                        dr_cut = truth_range

                    psnr_cut = compute_psnr(truth_cut, ae_cut, data_range=dr_cut)
                    msssim_cut = compute_msssim(torch.tensor(cubic_interpolate_along_axis(truth_cut, 161, axis=2), dtype=torch.float64), torch.tensor(cubic_interpolate_along_axis(ae_cut, 161, axis=2), dtype=torch.float64))

                    max_truth_idx_cut = np.nanargmax(truth_cut, axis=1)
                    ecs_truth_cut = depth_cut[max_truth_idx_cut]
                    max_ae_idx_cut = np.nanargmax(ae_cut, axis=1)
                    ecs_pred_cut = depth_cut[max_ae_idx_cut]

                    ae_rmse_cut = np.sqrt(np.mean((truth_cut - ae_cut) ** 2))
                    ae_ecs_rmse_cut = np.sqrt(np.mean((ecs_truth_cut - ecs_pred_cut) ** 2))
                    mae_cut = np.mean(np.abs(truth_cut - ae_cut))

                    min_max_idx_truth_cut = get_min_max_idx(truth_cut, pad=False)
                    min_max_idx_ae_cut = get_min_max_idx(ae_cut, pad=False)
                    mean_error_n_min_max_cut = np.mean(np.abs(np.sum(min_max_idx_truth_cut, axis=1) - np.sum(min_max_idx_ae_cut, axis=1)))
                    F1_score_cut = get_f1_score(min_max_idx_truth_cut, min_max_idx_ae_cut)
                    f1_score_cut = np.mean(F1_score_cut)

                    b_cut, a_cut = butter(N=2, Wn=0.107, btype='low', analog=False)
                    filtered_ae_cut = filtfilt(b_cut, a_cut, ae_cut, axis=1)
                    filtered_F1_score_cut = np.mean(get_f1_score(min_max_idx_truth_cut, get_min_max_idx(filtered_ae_cut, pad=False)))


                    # model_metrics[f'{model}'] = model_metrics.get(f'{model}', {})
                    # model_metrics[f'{model}'][cr] = {
                    #     'RMSE': ae_rmse_cut,
                    #     'PSNR': psnr_cut,
                    #     'MS-SSIM': msssim_cut,
                    #     'ECS': ae_ecs_rmse_cut,
                    #     'MAE': mae_cut,
                    #     'mean_error_n_min_max': mean_error_n_min_max_cut,
                    #     'F1_score': f1_score_cut,
                    #     'Filtered_F1_score': filtered_F1_score_cut,
                    #     'R2_score': 1 - (np.sum((truth_cut - ae_cut) ** 2) / np.sum((truth_cut - np.mean(truth_cut)) ** 2)),
                    # }

                    model_metrics[f'cutted_{model}'] = model_metrics.get(f'cutted_{model}', {})
                    model_metrics[f'cutted_{model}'][cr] = {
                        'RMSE': ae_rmse_cut,
                        'PSNR': psnr_cut,
                        'MS-SSIM': msssim_cut,
                        'ECS': ae_ecs_rmse_cut,
                        'MAE': mae_cut,
                        'mean_error_n_min_max': mean_error_n_min_max_cut,
                        'F1_score': f1_score_cut,
                        'Filtered_F1_score': filtered_F1_score_cut,
                        'R2_score': 1 - (np.sum((truth_cut - ae_cut) ** 2) / np.sum((truth_cut - np.mean(truth_cut)) ** 2)),
                    }
                except Exception:
                    if verbose:
                        print(f'    Skipping cutted metrics for {model} @ CR {cr} due to error')

    # Build data_dict (best/worst selected slices)
    data_dict = {}
    t, lat, lon = 15, 101, 81
    metrics_to_check = ['RMSE', 'F1_score', 'ECS', 'R2']

    def get_best_worst_random(test_ssp_truth, ae_ssp_test, depth_array, metric_name):
        if metric_name == 'RMSE':
            errors = np.sqrt(np.mean((test_ssp_truth - ae_ssp_test) ** 2, axis=1))
            score = -errors
        elif metric_name == 'MAE':
            errors = np.mean(np.abs(test_ssp_truth - ae_ssp_test), axis=1)
            score = -errors
        elif metric_name == 'ECS':
            max_truth_idx = np.nanargmax(test_ssp_truth, axis=1)
            max_pred_idx = np.nanargmax(ae_ssp_test, axis=1)
            ecs_truth = depth_array[max_truth_idx]
            ecs_pred = depth_array[max_pred_idx]
            errors = np.abs(ecs_truth - ecs_pred)
            score = -errors
        elif metric_name == 'R2':
            num = np.sum((test_ssp_truth - ae_ssp_test) ** 2, axis=1)
            den = np.sum((test_ssp_truth - np.mean(test_ssp_truth, axis=1, keepdims=True)) ** 2, axis=1)
            score = 1 - num / den
        elif metric_name == 'F1_score':
            min_max_idx_truth = get_min_max_idx(test_ssp_truth, pad=False)
            min_max_idx_ae = get_min_max_idx(ae_ssp_test, pad=False)
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
                't_lat': ((t0, lat0), test_ssp_truth[t0, :, lat0, :], ae_ssp_test[t0, :, lat0, :]),
                't_lat_lon': ((t0, lat0, lon0), test_ssp_truth[t0, :, lat0, lon0], ae_ssp_test[t0, :, lat0, lon0])
            }

        return {'best': extract(unravel(best_idx)), 'worst': extract(unravel(worst_idx))}

    if verbose:
        print('Selecting best/worst examples for each model')
    for model, cr_dict in outputs.items():
        data_dict[model] = {}
        for cr, ae_ssp_test in cr_dict.items():
            if verbose:
                print(f'  Selecting examples for {model} @ CR {cr}')
            data_dict[model][cr] = {}
            for metric in metrics_to_check:
                data_dict[model][cr][metric] = get_best_worst_random(test_ssp_truth, ae_ssp_test, depth_array, metric)
            data_dict[model][cr]['selected'] = {
                't_lat': ((t, lat), test_ssp_truth[t, :, lat, :], ae_ssp_test[t, :, lat, :]),
                't_lat_lon': ((t, lat, lon), test_ssp_truth[t, :, lat, lon], ae_ssp_test[t, :, lat, lon])
            }

    # Save outputs
    out_dir = Path(out_pickle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'icaspp_model_metrics_bis.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)
    with open(out_dir / 'icaspp_data_dict_bis.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Saved model_metrics and data_dict to {out_dir}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dm-mlic-pkl', default='/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_4_157_196_256.pkl')
    p.add_argument('--dm-cae-pkl', default='/Odyssey/private/o23gauvr/code/FASCINATION/pickle/dm_enatl_mean_std_along_depth_4_157_240_240.pkl')
    p.add_argument('--mlic-base-dir', default='/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/MLIC++')#'/Odyssey/private/o23gauvr/code/MLIC/experiments/keep')
    p.add_argument('--other-ckpt-base', default='/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/CAE')
    p.add_argument('--out-pickle-dir', default='/Odyssey/private/o23gauvr/code/FASCINATION/pickle')
    p.add_argument('--device', default='cuda')
    p.add_argument('--psnr-range-mode', choices=['truth', 'global', 'per', 'value'], default='truth', help='How to compute PSNR data range: truth (range from truth), global (range across truth+outputs), per (per-model range), value (explicit value)')
    p.add_argument('--psnr-range-value', type=float, default=None, help='Explicit data range to use when --psnr-range-mode value is selected')
    p.add_argument('--verbose', action='store_true', default=True, help='Enable verbose prints and progress bars')
    p.add_argument('--unique-name', action='store_true', default=True, help='Use unique names for MLIC++ models (no hyperparam details)')
    return p.parse_args()


def main():
    args = parse_args()
    compute_and_save(
        args.dm_mlic_pkl,
        args.dm_cae_pkl,
        args.mlic_base_dir,
        args.other_ckpt_base,
        args.out_pickle_dir,
        device=args.device,
        psnr_range_mode=args.psnr_range_mode,
        psnr_range_value=args.psnr_range_value,
        verbose=args.verbose,
        unique_name=args.unique_name
    )


if __name__ == '__main__':
    main()

#python src/compute_metrics.py --device cpu 
# pickle_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# # Define the file path for the pickle file
# pickle_file = pickle_dir / f"model_outputs_best_worst_ntbk.pkl"

# # Save the model_metrics dictionary
# with open(pickle_file, "wb") as f:
#     pickle.dump(data_dict, f)

# print(f"model_metrics dictionary saved to {pickle_file}")
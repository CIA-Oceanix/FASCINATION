import os


import sys
import json


running_path = "/Odyssey/private/o23gauvr/code/"
os.chdir(running_path)
sys.path.insert(0,running_path)

import pickle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt
import math

from scipy.linalg import qr
from torch.func import jacrev

from typing import Dict, List, Optional, Tuple


import re

from tqdm import tqdm 

from datetime import datetime

from scipy.interpolate import CubicSpline, PchipInterpolator
from MLIC.MLIC.utils.utils import Config
from MLIC.MLIC.models import MLICPlusPlus
from MLIC.MLIC.utils.lr_scheduler import CosineWithFloor, SmoothCosineDecay

from pathlib import Path
from FASCINATION.src.utils import unorm_ssp_arr_3D, norm_ssp_arr_3D, get_cfg_from_ckpt_path, load_model


def compute_total_bits(out_net):
    return sum(torch.log(likelihoods).sum() / (-math.log(2))
              for likelihoods in out_net['likelihoods'].values()).item()


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu", dm=None, batch=None):
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    
    # Detect model type from checkpoint
    if 'MLIC' in checkpoint_path:
        # MLIC format
        cfg = parse_mlic_config(Path(checkpoint_path))
        model = MLICPlusPlus(config=Config(cfg))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
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
    
    return model, checkpoint, model_type, cfg





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
        
        # N = cfg['N']
        # M = cfg['M']

    #     if N is not None and M is not None:
    #         print(f"Extracted from {model_path}: N={N}, M={M}")
    #         return N, M
    #     else:
    #         print(f"Warning: Could not extract N and M from {model_path}")
    #         return None, None
            
    except Exception as e:
        print(f"Error parsing config file in {ckpt_path}: {e}")
        return None



def parse_mlic_config(ckpt_path):
    """
    Parse experiment_config.log file to extract N and M values.
    
    Args:
        ckpt_path (str): PosixPath to checkpoint 
        
    Returns:
        tuple: (N, M) values or (None, None) if parsing fails
    """

    config_file = list((ckpt_path.parent.parent).rglob("experiment_config.log"))
    if not config_file:
        print(f"Warning: experiment_config.log not found in {ckpt_path}")
        return None

    config_file = sorted(config_file)[0]

    try:
        with open(config_file, 'r') as f:
            content = f.read()

        def _extract_section(section_name: str) -> str:
            header_re = re.compile(
                rf"{re.escape(section_name)}\n-+\n",
                flags=re.MULTILINE,
            )
            start_match = header_re.search(content)
            if not start_match:
                return ""

            next_header_re = re.compile(r"\n[A-Z][A-Z _]+:\n-+\n", flags=re.MULTILINE)
            next_match = next_header_re.search(content, start_match.end())
            end_idx = next_match.start() if next_match else len(content)
            return content[start_match.end():end_idx].strip()

        def _parse_key_value_blocks(block: str) -> Dict[str, str]:
            if not block:
                return {}
            entry_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*", flags=re.MULTILINE)
            matches = list(entry_re.finditer(block))
            parsed = {}
            for i, match in enumerate(matches):
                key = match.group(1)
                value_start = match.end()
                value_end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
                parsed[key] = block[value_start:value_end].strip()
            return parsed

        def _safe_eval_model_value(raw_value: str):
            value = raw_value.replace("<class 'torch.nn.modules.activation.", "nn.").replace("'>", "")
            try:
                return eval(value, {"__builtins__": {}}, {"nn": nn, "np": np, "True": True, "False": False, "None": None})
            except Exception:
                return raw_value

        model_block = _extract_section("MODEL CONFIGURATION:")
        model_cfg_raw = _parse_key_value_blocks(model_block)
        cfg = {k: _safe_eval_model_value(v) for k, v in model_cfg_raw.items()}

        dm_block = _extract_section("DATAMODULE PARAMETERS:")
        dm_cfg_all = _parse_key_value_blocks(dm_block)
        if dm_cfg_all and "dl_kw" in dm_cfg_all and "test_shape" in dm_cfg_all:
            dm_keys = list(dm_cfg_all.keys())
            start_idx = dm_keys.index("dl_kw")
            end_idx = dm_keys.index("test_shape")
            selected_dm_keys = dm_keys[start_idx:end_idx + 1]
            datamodule_info = {k: dm_cfg_all[k] for k in selected_dm_keys}
        else:
            datamodule_info = {}

        loss_block = _extract_section("LOSS FUNCTION PARAMETERS:")
        loss_function_parameters = _parse_key_value_blocks(loss_block)

        cfg["datamodule_info"] = datamodule_info
        cfg["loss_function_parameters"] = loss_function_parameters

        return cfg
        
    except Exception as e:
        print(f"Error parsing config file in {ckpt_path}: {e}")
        return None


def load_mlic(ssp_truth_unorm, state, norm_stats, mlic_path, model, cfg , device=None, batch_size=4):

    if device is None:
        device = next(model.parameters()).device
    device = str(device)

    #ssp_truth = unorm_ssp_arr_3D(ssp_truth, state["test_norm_stats"])

    if norm_stats=="train":
        if "train_norm_stats" in state:
            model_norm_stats = state["train_norm_stats"]


    elif norm_stats=="test":
        #model_norm_stats = state.get("test_norm_stats", dm_test_norm)
        if "test_norm_stats" in state:
            model_norm_stats = state["test_norm_stats"]


    ssp_truth = norm_ssp_arr_3D(ssp_truth_unorm, model_norm_stats).astype(np.float32)
    ssp_tensor = torch.from_numpy(ssp_truth).to(dtype=torch.float32, device=device)

    x_hat_batches = []
    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in range(0, ssp_tensor.shape[0], batch_size):
            end = min(start + batch_size, ssp_tensor.shape[0])
            batch = ssp_tensor[start:end].to(device)
            rv_batch = model(batch)
            x_hat_batches.append(rv_batch["x_hat"].detach().cpu())
            del batch, rv_batch

    ssp_ae_tensor = torch.cat(x_hat_batches, dim=0)

    
    # should_filter_output = mlic_path is not None and any(
    #     keyword in str(mlic_path) for keyword in ["rmse", "ecs", "best_checkpoint_loss"]
    # )
    if not cfg.get('output_low_band_filter_use', False):

        x_hat_np = ssp_ae_tensor.detach().cpu().numpy()
        b, a = butter(N=2, Wn=0.107, btype="low", analog=False)
        x_hat_np = filtfilt(b, a, x_hat_np, axis=1).astype(x_hat_np.dtype)
        ssp_ae_tensor = torch.from_numpy(x_hat_np).to(ssp_ae_tensor.device)



    #compressed_size_bits = compute_total_bits(rv_batch)


    # original_size_bits = ssp_truth_norm.nbytes * 8
    # cr = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else float('inf')
    # N, _, H, W = ssp_tensor.size()
    # num_pixels = N * H * W
    # bpe = compute_bpe(rv_batch, num_pixels=num_pixels)

    # ssp_ae = ssp_ae_tensor.cpu().numpy().astype(np.float32)
    # ssp_ae = unorm_ssp_arr_3D(ssp_ae, model_norm_stats)


    # should_filter_output = any(keyword in mlic_path for keyword in ['rmse', 'ecs', 'best_checkpoint_loss'])
    # if should_filter_output:
    #     b, a = butter(N=2, Wn=0.107, btype='low', analog=False)
    #     ssp_ae = filtfilt(b, a, ssp_ae, axis=1).astype(ssp_ae.dtype)

    # ssp_ae_da = xr.DataArray(
    #     ssp_ae,
    #     coords=ssp_truth_da.coords,
    #     dims=ssp_truth_da.dims,
    #     name='ssp_reconstructed'
    # )

    return ssp_tensor, ssp_ae_tensor



def psi(model, y_hat):
    x_hat = model.g_s(y_hat)
    x_hat = x_hat[:,:C]
    x_hat = model._apply_output_low_band_filter(x_hat)
    return x_hat



def build_temporal_observation_mask(time_values, n_days, time_start=None):
    """Return a boolean mask selecting a contiguous temporal window."""
    if n_days <= 0:
        raise ValueError("n_days must be positive")

    time_values = np.asarray(time_values)
    if time_values.ndim != 1 or time_values.size == 0:
        raise ValueError("time_values must be a non-empty 1D array")

    if time_start is None:
        start_idx = 0
    else:
        start_time = np.datetime64(time_start)
        start_idx = int(np.searchsorted(time_values, start_time, side="left"))
        if start_idx >= len(time_values):
            raise ValueError(f"time_start={time_start} is after the last available time value")

    end_idx = min(start_idx + int(n_days), len(time_values))
    mask = np.zeros(len(time_values), dtype=bool)
    mask[start_idx:end_idx] = True
    return mask


def apply_eval_postprocessing(x_hat, cfg={}, crop_idx=None, mlic_path=None):
    """Apply evaluation-only cropping/filtering to a reconstruction."""
    if crop_idx is not None:
        x_hat = x_hat[..., crop_idx, crop_idx]

    # should_filter_output = mlic_path is not None and any(
    #     keyword in str(mlic_path) for keyword in ["rmse", "ecs", "best_checkpoint_loss"]
    # )
    if not cfg.get('output_low_band_filter_use', False):

        x_hat_np = x_hat.detach().cpu().numpy()
        b, a = butter(N=2, Wn=0.107, btype="low", analog=False)
        x_hat_np = filtfilt(b, a, x_hat_np, axis=1).astype(x_hat_np.dtype)
        x_hat = torch.from_numpy(x_hat_np).to(x_hat.device)

    return x_hat


def evaluate_full_span_reconstruction(x_hat, truth, state, norm_stats, cfg={}, crop_idx=None, mlic_path=None):
    """Compute full-span metrics after the same post-processing used at eval time."""
    x_hat_eval = apply_eval_postprocessing(x_hat.detach(), cfg=cfg, crop_idx=crop_idx, mlic_path=mlic_path)
    truth_eval = truth.detach()

    if crop_idx is not None:
        truth_eval = truth_eval[..., crop_idx, crop_idx]

    x_hat_eval = unorm_ssp_arr_3D(x_hat_eval.detach().cpu().numpy(), state[norm_stats + "_norm_stats"])
    truth_eval = unorm_ssp_arr_3D(truth_eval.detach().cpu().numpy(), state["test_norm_stats"])

    diff_unorm = x_hat_eval - truth_eval
    time_rmse = np.sqrt(np.mean(diff_unorm ** 2, axis=(1, 2, 3)))
    rec_spatial_std_per_day = np.std(diff_unorm, axis=(1, 2, 3))

    return {
        "rec_rmse_mean": float(np.sqrt(np.mean(diff_unorm ** 2))),
        "rec_rmse_std": float(np.std(time_rmse)),
        "mae": float(np.mean(np.abs(diff_unorm))),
        "time_rmse": time_rmse,
        "depth_time_rmse": np.sqrt(np.mean(diff_unorm ** 2, axis=(2, 3))),
        "rec_spatial_std_mean": float(np.mean(rec_spatial_std_per_day)),
        "rec_spatial_std_std": float(np.std(rec_spatial_std_per_day)),
        "rec_spatial_std_per_day": rec_spatial_std_per_day,
    }


def make_spatial_observation_mask(sample, n_obs, seed=42):
    """Build a spatial observation mask for one [B, C, H, W] sample."""
    mask = torch.zeros_like(sample[:, 0, :, :])
    generator = torch.Generator(device=mask.device)
    generator.manual_seed(seed)
    numel = mask.numel()
    n_obs = min(int(n_obs), numel)
    idx = torch.randperm(numel, generator=generator, device=mask.device)[:n_obs]
    mask.view(-1)[idx] = 1
    return mask.unsqueeze(1).repeat(1, sample.shape[1], 1, 1)


def reconstruct_full_series_from_selected_days(full_series, selected_indices, selected_reconstructions):
    """Fill a full time series by carrying the nearest selected reconstruction in time."""
    selected_indices = np.asarray(selected_indices)
    full_recon = full_series.clone()

    for t_idx in range(full_series.shape[0]):
        nearest_pos = int(np.argmin(np.abs(selected_indices - t_idx)))
        nearest_time = int(selected_indices[nearest_pos])
        full_recon[t_idx] = selected_reconstructions[nearest_time][0].squeeze(0)

    return full_recon


def plot_full_span_results(baseline_metrics, cadence_results, time_values, selected_indices_by_cadence, output_path=None):
    """Plot baseline vs optimized full-span metrics for each observation cadence."""
    cadences = list(cadence_results.keys())
    fig, axs = plt.subplots(len(cadences), 3, figsize=(20, 5 * len(cadences)), squeeze=False)

    time_values = np.asarray(time_values)
    time_axis = np.arange(len(time_values))

    for row, cadence in enumerate(cadences):
        cadence_payload = cadence_results[cadence]

        # Accept either a direct metrics dict, or a nested dict keyed by config.
        if isinstance(cadence_payload, dict) and "time_rmse" in cadence_payload:
            metrics = cadence_payload
        elif isinstance(cadence_payload, dict) and len(cadence_payload) > 0:
            first_key = next(iter(cadence_payload.keys()))
            first_payload = cadence_payload[first_key]
            if isinstance(first_payload, dict) and "metrics" in first_payload:
                metrics = first_payload["metrics"]
            elif isinstance(first_payload, dict) and "time_rmse" in first_payload:
                metrics = first_payload
            else:
                raise ValueError(
                    f"Unsupported cadence_results payload for '{cadence}'. "
                    "Expected metrics or {'metrics': ...} entries."
                )
        else:
            raise ValueError(f"Unsupported cadence_results payload for '{cadence}'.")

        selected_indices = selected_indices_by_cadence[cadence]

        ax0, ax1, ax2 = axs[row]

        ax0.plot(time_axis, baseline_metrics["time_rmse"], label="baseline", alpha=0.75)
        ax0.plot(time_axis, metrics["time_rmse"], label="optimized", lw=2.0)
        for idx in selected_indices:
            ax0.axvline(idx, color="k", alpha=0.12, lw=1)
        ax0.set_title(f"Time RMSE - cadence {cadence}")
        ax0.set_ylabel("RMSE")
        ax0.legend()
        ax0.grid(True, alpha=0.2)

        improvement = baseline_metrics["time_rmse"] - metrics["time_rmse"]
        ax1.plot(time_axis, improvement, color="tab:green", lw=2)
        for idx in selected_indices:
            ax1.axvline(idx, color="k", alpha=0.12, lw=1)
        ax1.axhline(0.0, color="black", lw=1, alpha=0.5)
        ax1.set_title(f"RMSE improvement over baseline - cadence {cadence}")
        ax1.set_ylabel("Baseline - optimized")
        ax1.grid(True, alpha=0.2)

        heatmap = metrics["depth_time_rmse"].T
        im = ax2.imshow(heatmap, aspect="auto", origin="lower", cmap="magma")
        for idx in selected_indices:
            ax2.axvline(idx, color="cyan", alpha=0.18, lw=1)
        ax2.set_title(f"Depth-time RMSE - cadence {cadence}")
        ax2.set_xlabel("Time index")
        ax2.set_ylabel("Depth index")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300)
    return fig



def initialize_with_nearest_neighbor(model, x_train, obs, mask, k=5, encode_batch_size=8):
   
    """
    Initialize latent variables for optimization using a nearest-neighbor heuristic.

    For each (partially observed) target sample in `obs`, finds the k most similar
    training samples in `x_train` — similarity computed in the *observation space*,
    restricted to the dimensions marked available in `mask` — and initializes that
    sample's latent code as the average of the corresponding precomputed training
    latents (from `phi_v2`). Samples with no observed dimensions fall back to the
    mean latent code over all of `x_train`.

    Args:
        model: compression model exposing `phi_v2` for latent encoding.
        x_train (Tensor): training inputs, shape (N_train, ...).
        obs (Tensor): partially observed target samples, shape (N, D).
        mask (Tensor): boolean/0-1 mask of shape (N, D) indicating which
            dimensions of each row in `obs` are observed.
        k (int): number of nearest training neighbors to average over.

    Returns:
        Tensor: initial latent codes for each sample in `obs`, with
            `requires_grad=True`, ready to be optimized.
    """

    model.eval()
    model_device = next(model.parameters()).device

    x_train_cpu = x_train.detach().cpu()
    obs_cpu = obs.detach().cpu()
    mask_cpu = mask.detach().cpu()

    with torch.no_grad():
        train_y_hat_batches = []
        encode_batch_size = max(1, int(encode_batch_size))
        for start in range(0, x_train_cpu.shape[0], encode_batch_size):
            end = min(start + encode_batch_size, x_train_cpu.shape[0])
            x_batch = x_train_cpu[start:end].to(model_device)
            train_y_hat_batches.append(model(x_batch)["y_hat"].detach().cpu())
            del x_batch

        train_y_hat = torch.cat(train_y_hat_batches, dim=0)

        y_initial = []
        for i in range(len(obs_cpu)):
            available_indices = mask_cpu[i].bool()
            if available_indices.sum() > 0:
                similarities = torch.sum(
                    (x_train_cpu[:, available_indices] - obs_cpu[i, available_indices]) ** 2,
                    dim=1,
                )
                k_eff = min(int(k), int(similarities.shape[0]))
                nn_indices = similarities.topk(k_eff, largest=False).indices
                y_initial.append(train_y_hat[nn_indices].mean(dim=0))
            else:
                y_initial.append(train_y_hat.mean(dim=0))

        y = torch.stack(y_initial).to(model_device)

    return y.requires_grad_(True)




def spline_reconstruct_from_sparse(obs, method="pchip"):
    """
    Reconstruct x_hat[B, D, W, H] from sparse obs[B, D, W, H],
    where obs contains zeros at positions without data.

    Parameters
    ----------
    obs : np.ndarray
        Sparse input of shape [B, D, W, H]
    method : {"cubic", "pchip", "linear"}
        Interpolation method along D dimension.

    Returns
    -------
    x_hat : np.ndarray
        Reconstructed array of shape [B, D, W, H]
    """

    B, D, W, H = obs.shape
    x_hat = np.zeros_like(obs)

    x_full = np.arange(D)

    for b in range(B):
        for w in range(W):
            for h in range(H):

                y_line = obs[b, :, w, h]

                # find non-zero positions (valid samples)
                idx = np.where(y_line != 0)[0]

                # case 1: no samples -> leave zeros
                if len(idx) == 0:
                    continue

                # case 2: only 1 sample -> constant interpolation
                if len(idx) == 1:
                    x_hat[b, :, w, h] = y_line[idx[0]]
                    continue

                # get the valid sample values
                values = y_line[idx]

                if method == "cubic":
                    f = CubicSpline(idx, values, bc_type='natural')
                elif method == "pchip":
                    f = PchipInterpolator(idx, values)
                elif method == "linear":
                    from scipy.interpolate import interp1d
                    f = interp1d(idx, values, kind='linear',
                                    fill_value="extrapolate")
                else:
                    # fallback to numpy linear interpolation
                    x_hat[b, :, w, h] = np.interp(x_full, idx, values)
                    #raise ValueError("Unknown interpolation method.")


                x_hat[b, :, w, h] = f(x_full)




    return x_hat


def broadcast_spatial_mask(mask_2d: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a [H, W] or [B, H, W] mask to match a [B, C, H, W] tensor."""
    if mask_2d.dim() == 2:
        mask_2d = mask_2d.unsqueeze(0)
    if mask_2d.dim() != 3:
        raise ValueError(f"Expected a 2D or 3D spatial mask, got shape {tuple(mask_2d.shape)}")
    if reference.dim() != 4:
        raise ValueError(f"Expected a reference tensor with shape [B, C, H, W], got {tuple(reference.shape)}")
    return mask_2d.unsqueeze(1).expand(reference.shape[0], reference.shape[1], -1, -1)


def compute_latent_objective_terms(
    model,
    y_hat: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    loss_on: str = "obs",
    crop_idx=None,
):
    """Return torch-only reconstruction terms for latent optimization."""
    x_hat = psi(model, y_hat)
    x_hat = x_hat[..., :obs.shape[-2], :obs.shape[-1]] if x_hat.shape[-2:] != obs.shape[-2:] else x_hat

    x_hat_for_loss = x_hat
    obs_for_loss = obs
    mask_for_loss = mask

    if crop_idx is not None:
        x_hat_for_loss = x_hat_for_loss[..., crop_idx, crop_idx]
        obs_for_loss = obs_for_loss[..., crop_idx, crop_idx]
        mask_for_loss = mask_for_loss[..., crop_idx, crop_idx]

    reconstruction_diff = x_hat_for_loss - obs_for_loss
    obs_diff = mask_for_loss * reconstruction_diff

    if loss_on == "obs":
        data_loss = obs_diff.pow(2).mean()
    elif loss_on == "full":
        data_loss = reconstruction_diff.pow(2).mean()
    else:
        raise ValueError(f"Unsupported loss_on='{loss_on}'")

    y_rec = model(x_hat)["y_hat"]
    latent_consistency = F.mse_loss(y_rec, y_hat)

    return {
        "x_hat": x_hat,
        "x_hat_for_loss": x_hat_for_loss,
        "obs_for_loss": obs_for_loss,
        "mask_for_loss": mask_for_loss,
        "reconstruction_diff": reconstruction_diff,
        "obs_diff": obs_diff,
        "data_loss": data_loss,
        "latent_consistency": latent_consistency,
        "loss": data_loss,
    }


def learn_sensor_mask_with_latent_optimization(
    model,
    obs,
    state,
    cfg={},
    loss_on="obs",
    y_hat_init="0_filled",
    x_train=None,
    k=5,
    nn_encode_batch_size=8,
    spline_method="pchip",
    crop_idx=None,
    mlic_path=None,
    n_outer_steps=25,
    y_steps_per_outer=25,
    mask_lr=1e-2,
    y_lr=1e-2,
    lam=1.0,
    lam_budget=1.0,
    lam_entropy=1e-3,
    budget=5,
    device="cuda",
    verbose=True,
):
    """Alternate latent optimization with differentiable mask-logit updates."""
    model.eval()
    obs = obs.to(device)

    if obs.dim() != 4:
        raise ValueError(f"Expected obs with shape [B, C, H, W], got {tuple(obs.shape)}")

    batch_size, channels, height, width = obs.shape
    mask_logits = torch.zeros((batch_size, height, width), device=device, requires_grad=True)
    mask_optimizer = torch.optim.Adam([mask_logits], lr=mask_lr)

    spatial_mask = torch.ones((batch_size, 1, height, width), device=device)

    if y_hat_init == "nn" and x_train is not None:
        y_hat = initialize_with_nearest_neighbor(
            model,
            x_train,
            obs,
            spatial_mask.expand(batch_size, channels, height, width),
            k=k,
            encode_batch_size=nn_encode_batch_size,
        )
    elif y_hat_init == "train_mean_profile" and x_train is not None:
        fill = x_train.mean(dim=(0, 2, 3), keepdim=True).to(device=device, dtype=obs.dtype)
        x_filled = obs + 0.0
        x_filled = x_filled * 0 + fill
        with torch.no_grad():
            y_hat = model(x_filled)["y_hat"]
    else:
        if y_hat_init == "obs_mean_profile":
            fill = obs.mean(dim=(0, 2, 3), keepdim=True)
            x_filled = fill.expand_as(obs).clone()
        elif y_hat_init == "spline_interp":
            obs_np = obs.detach().cpu().numpy()
            obs_filled_np = spline_reconstruct_from_sparse(obs_np, method=spline_method)
            x_filled = torch.tensor(obs_filled_np, device=device, dtype=obs.dtype)
        else:
            x_filled = torch.zeros_like(obs)

        with torch.no_grad():
            y_hat = model(x_filled)["y_hat"]

    y_hat = y_hat.detach().clone().requires_grad_(True)
    y_optimizer = torch.optim.Adam([y_hat], lr=y_lr)

    history = {
        "loss": [],
        "data_loss": [],
        "latent_consistency": [],
        "budget_penalty": [],
        "entropy_penalty": [],
        "mask_sum": [],
        "obs_diff": [],
        "obs_rmse": [],
        "relative_obs_rmse": [],
        "reconstruction_diff": [],
        "rec_rmse": [],
        "relative_rec_rmse": [],
        "lr": [],
    }

    best_loss = float("inf")
    best_y_hat = None
    best_mask = None
    initial_terms = compute_latent_objective_terms(
        model,
        y_hat,
        obs,
        torch.sigmoid(mask_logits).unsqueeze(1).expand(batch_size, channels, height, width),
        loss_on=loss_on,
        crop_idx=crop_idx,
    )
    xhat_0 = initial_terms["x_hat"].detach().clone()

    for outer_step in range(n_outer_steps):
        soft_mask = torch.sigmoid(mask_logits).unsqueeze(1).expand(batch_size, channels, height, width)

        for _ in range(max(1, int(y_steps_per_outer))):
            y_optimizer.zero_grad()
            terms = compute_latent_objective_terms(
                model,
                y_hat,
                obs,
                soft_mask,
                loss_on=loss_on,
                crop_idx=crop_idx,
            )
            loss = terms["data_loss"] + lam * terms["latent_consistency"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_([y_hat], max_norm=1.0)
            y_optimizer.step()

        mask_optimizer.zero_grad()
        soft_mask = torch.sigmoid(mask_logits).unsqueeze(1).expand(batch_size, channels, height, width)
        terms = compute_latent_objective_terms(
            model,
            y_hat,
            obs,
            soft_mask,
            loss_on=loss_on,
            crop_idx=crop_idx,
        )
        mask_2d = torch.sigmoid(mask_logits)
        budget_target = torch.as_tensor(float(budget), device=device, dtype=mask_2d.dtype)
        budget_penalty = (mask_2d.sum(dim=(-2, -1)) - budget_target).pow(2).mean()
        entropy_penalty = -(
            mask_2d * torch.log(mask_2d.clamp_min(1e-6))
            + (1.0 - mask_2d) * torch.log((1.0 - mask_2d).clamp_min(1e-6))
        ).mean()

        mask_loss = terms["data_loss"] + lam * terms["latent_consistency"] + lam_budget * budget_penalty + lam_entropy * entropy_penalty
        mask_loss.backward()
        mask_optimizer.step()

        current_loss = float(mask_loss.item())
        history["loss"].append(current_loss)
        history["data_loss"].append(float(terms["data_loss"].item()))
        history["latent_consistency"].append(float(terms["latent_consistency"].item()))
        history["budget_penalty"].append(float(budget_penalty.item()))
        history["entropy_penalty"].append(float(entropy_penalty.item()))
        history["mask_sum"].append(float(mask_2d.sum().item()))

        obs_diff_eval = terms["obs_diff"].detach()
        reconstruction_diff_eval = terms["reconstruction_diff"].detach()
        obs_rmse = float(torch.sqrt(obs_diff_eval.pow(2).mean()).item())
        rec_rmse = float(torch.sqrt(reconstruction_diff_eval.pow(2).mean()).item())
        budget_value = max(1, int(budget))

        history["obs_diff"].append(float(obs_diff_eval.pow(2).mean().item()))
        history["obs_rmse"].append(obs_rmse)
        history["relative_obs_rmse"].append(obs_rmse / budget_value)
        history["reconstruction_diff"].append(float(reconstruction_diff_eval.pow(2).mean().item()))
        history["rec_rmse"].append(rec_rmse)
        history["relative_rec_rmse"].append(rec_rmse / budget_value)
        history["lr"].append(float(y_optimizer.param_groups[0]["lr"]))

        if current_loss < best_loss:
            best_loss = current_loss
            best_y_hat = y_hat.detach().clone()
            best_mask = torch.sigmoid(mask_logits).detach().clone()

        if verbose and (outer_step % 10 == 0 or outer_step == n_outer_steps - 1):
            print(
                f"[mask-opt] step {outer_step + 1}/{n_outer_steps} "
                f"loss={current_loss:.6e} data={history['data_loss'][-1]:.6e} "
                f"budget={history['budget_penalty'][-1]:.6e} entropy={history['entropy_penalty'][-1]:.6e}"
            )

    best_xhat = psi(model, best_y_hat)
    best_mask_hard = project_soft_mask_to_budget(best_mask, budget) if best_mask is not None else torch.zeros((batch_size, height, width), device=device, dtype=obs.dtype)
    return best_y_hat, best_xhat.detach(), xhat_0, best_mask_hard, history


def qr_pivot_sensor_candidates(train_profiles, budget, rank=10):
    """Return flat sensor indices from a POD basis followed by QR pivoting."""
    train_profiles = np.asarray(train_profiles)
    b,c,h,w = train_profiles.shape
    if train_profiles.ndim < 2:
        raise ValueError("train_profiles must have at least 2 dimensions")

    feature_dim = int(np.prod(train_profiles.shape[1:]))
    sample_count = train_profiles.shape[0]
    if sample_count == 0:
        raise ValueError("train_profiles must contain at least one sample")

    x = train_profiles.transpose(2,3,0,1).reshape(h*w,b*c)
    #x = train_profiles.reshape(sample_count, feature_dim).T
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    r_eff = max(1, min(int(rank), u.shape[1]))
    basis = u[:, :r_eff]

    from scipy.linalg import qr

    _, _, pivots = qr(basis.T, pivoting=True)
    budget = max(1, min(int(budget), pivots.shape[0]))
    return pivots[:budget].tolist()


def qr_over_latent_jacobian(y_hat, budget):

    def decoder(latent):
        return psi(model, latent)


    x_hat = psi(model, y_hat)
    C, H, W = x_hat.shape[1:]
    jacobian = jacrev(decoder)(y_hat)
    latent_dim = int(np.prod(jacobian.shape[5:]))
    J = jacobian.squeeze(0)                   # remove batch
    J = J.reshape(C, H, W, latent_dim)
    J_space = J.permute(1,2,0,3).reshape(H*W, C*latent_dim)
    _, _, piv = qr(J_space.T.cpu().numpy(), pivoting=True)

    candidate_indices = piv[:budget]
    return candidate_indices


def compute_sensor_score(
    model,
    x_train,
    crop_idx=None,
    batch_size=8,
    spatial_block=64,
    device="cuda",
):
    """
    Estimate decoder sensitivity for placing SSP profilers.

    Returns
    -------
    score : (H,W) tensor on CPU
        Large score = good sensor location.
    """

    model.eval()

    ##############################################
    # Encode training samples
    ##############################################

    y_list = []

    with torch.no_grad():
        for i in range(0, len(x_train), batch_size):

            xb = x_train[i:i+batch_size].to(device)

            y = model(xb)["y_hat"]

            y_list.append(y)

    y_hat = torch.cat(y_list, dim=0).detach()

    y_hat.requires_grad_(True)

    ##############################################
    # Decode
    ##############################################

    x_hat = psi(model, y_hat)

    H0, W0 = x_hat.shape[-2:]

    if crop_idx is not None:
        x_hat = x_hat[..., crop_idx, crop_idx]

    B, C, H, W = x_hat.shape

    score = torch.zeros(H * W, device='cpu')

    ##############################################
    # Flatten spatial dimension
    ##############################################

    profiles = (
        x_hat.permute(2,3,0,1)
             .reshape(H*W, B*C)
    )

    ##############################################
    # Process several locations together
    ##############################################

    for start in tqdm(range(0, H*W, spatial_block)):

        stop = min(start + spatial_block, H*W)

        scalar = profiles[start:stop].sum()

        grad = torch.autograd.grad(
            scalar,
            y_hat,
            retain_graph=True,
        )[0]

        gnorm = grad.pow(2).sum().sqrt()

        score[start:stop] = gnorm.to('cpu')

    ##############################################

    score = score.reshape(H, W).cpu()

    if crop_idx is not None:

        full = torch.zeros(H0, W0)

        full[crop_idx, crop_idx] = score

        score = full

    return score


def compute_sensor_score_fast(model, x_train, crop_idx=None, batch_size=8, device='cuda', n_probe=8):


    with torch.no_grad():
        y_hat_batches = []
        batch_size = max(1, int(batch_size))
        for start in range(0, x_train.shape[0], batch_size):
            end = min(start + batch_size, x_train.shape[0])
            x_batch = x_train[start:end].to(device)
            y_hat_batches.append(model(x_batch)['y_hat'].detach().clone().requires_grad_(True))
            del x_batch

    y_hat = torch.cat(y_hat_batches, dim=0).detach().clone().requires_grad_(True)

    x_hat = psi(model, y_hat)

    original_H, original_W = x_hat.shape[2:]

    if crop_idx is not None:
        x_hat = x_hat[..., crop_idx, crop_idx]

    B,C,H,W = x_hat.shape

    score = torch.zeros(H,W,device=x_hat.device)

    for _ in range(n_probe):

        r = torch.randn(B,C,H,W,device=x_hat.device)

        scalar = (x_hat*r).sum()

        grad = torch.autograd.grad(
            scalar,
            y_hat,
            retain_graph=True
        )[0]

        # distribute contribution using output projection
        local = (x_hat*r).abs().mean(dim=(0,1))

        score += local

    score = score/n_probe

    full_score = torch.zeros(original_H, original_W, device='cpu')

    full_score[crop_idx,crop_idx] = score

    return full_score


def build_mask_from_flat_indices(reference: torch.Tensor, flat_indices) -> torch.Tensor:
    """Create a binary [B, C, H, W] mask from flat spatial indices."""
    if reference.dim() != 4:
        raise ValueError(f"Expected reference tensor with shape [B, C, H, W], got {tuple(reference.shape)}")
    mask = torch.zeros((reference.shape[0], reference.shape[2], reference.shape[3]), device=reference.device, dtype=reference.dtype)
    indices = torch.as_tensor(flat_indices, device=reference.device, dtype=torch.long)
    if indices.numel() > 0:
        mask.view(-1)[indices] = 1.0
    return mask.unsqueeze(1).expand_as(reference)


def project_soft_mask_to_budget(mask_2d: torch.Tensor, budget: int) -> torch.Tensor:
    """Project a soft [B, H, W] mask to a hard top-k mask per batch item."""
    if mask_2d.dim() != 3:
        raise ValueError(f"Expected a mask with shape [B, H, W], got {tuple(mask_2d.shape)}")
    budget = max(0, int(budget))
    batch_size, height, width = mask_2d.shape
    flat = mask_2d.reshape(batch_size, -1)
    hard = torch.zeros_like(flat)
    topk = min(budget, flat.shape[1])
    if topk > 0:
        idx = torch.topk(flat, k=topk, dim=1, largest=True).indices
        hard.scatter_(1, idx, 1.0)
    return hard.view(batch_size, height, width)



def optimize_latent_from_observations(model, obs, mask, 
                                      state,
                                      cfg={},
                                      loss_on="obs",
                                      y_hat_init="0_filled",
                                      x_train=None,
                                      k = 5,
                                      nn_encode_batch_size=8,
                                      spline_method="pchip",
                                      crop_idx=None,
                                      mlic_path=None,
                                      n_steps=800, lr=1e-2, lam=1.0,
                                      device='cuda', verbose=True):
    """
    Optimize continuous hyper-latent z (before quant) to minimize:
       L(Z) = || mask * (psi(Z) - obs) ||_2^2 + lambda * || phi(psi(Z)) - Z ||_2^2
    where psi(Z) = psi_from_hyper(model, Z) and phi = h_a(g_a(.))
    Args:
      - model: your trained model (modules used above must exist)
      - obs: observed image tensor [B, C, H, W] (partial / sparse values)
      - mask: same shape as obs, binary 0/1 for observed pixels
      - y_hat_init: optional initial z (tensor matching model.h_a output)
      - n_steps, lr, lam: optimization hyperparams
    Returns:
      - best_z, best_xhat, history_losses
    """
    model.eval()
    obs = obs.to(device)
    mask = mask.to(device)
    n_obs = int(mask[:,0,:,:].sum())
    data_loss=0

    # prepare z variable
    if y_hat_init == "0_filled":
        fill = torch.zeros_like(obs) #torch.ones_like(obs)*obs.mean(axis=(0,2,3), keepdim=True) #torch.zeros_like(obs)
        x_filled = mask * obs + (1 - mask) * fill
        # reasonable init: encode a simple inpaint of obs (fill masked with zeros or mean)
    elif y_hat_init == "obs_mean_profile":
        fill = obs.mean(axis=(0,2,3), keepdim=True)
        x_filled = mask * obs + (1 - mask) * fill
    elif y_hat_init == "train_mean_profile":
        fill = x_train.mean(axis=(0,2,3), keepdim=True).to(device=obs.device, dtype=obs.dtype)
        x_filled = mask * obs + (1 - mask) * fill
    elif y_hat_init == "spline_interp":
        obs_np = (obs*mask).detach().cpu().numpy()
        obs_filled_np = spline_reconstruct_from_sparse(obs_np, method=spline_method)
        x_filled = torch.tensor(obs_filled_np).to(device)


    if y_hat_init in ["0_filled", "obs_mean_profile", "train_mean_profile", "spline_interp"]:

        with torch.no_grad():
            try:
                y_hat = model(x_filled)['y_hat']

            except Exception as e:
                # fallback: random small init
                raise e
                #z0 = torch.randn(1, *model.h_a(torch.zeros_like(x_filled)).shape[1:]).to(device) * 1e-2


    if y_hat_init == "nn":
        y_hat = initialize_with_nearest_neighbor(
            model,
            x_train,
            obs,
            mask,
            k=k,
            encode_batch_size=nn_encode_batch_size,
        )
        with torch.no_grad():
            x_filled = psi(model,y_hat)


    y_hat = y_hat.detach().clone().requires_grad_(True)



    optimizer = torch.optim.Adam([y_hat], lr=lr)
    # lr_scheduler = SmoothCosineDecay(
    #     optimizer,
    #     t_initial=1000,
    #     lr_min=1e-6,
    #     warmup_t=5,
    #     cycle_limit=100000,
    #     warmup_lr_init=optimizer.defaults['lr'],
    #     cycle_decay=0.5,
    # )
    lr_scheduler = CosineWithFloor(optimizer, T=10000, eta_min=1e-4)  # <-- Use the custom scheduler with a floor
    #lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=n_steps)  # <-- Use the custom scheduler with a floor

    best_loss = float('inf')
    best_y_hat = None
    history = {'loss':[], 'obs_diff':[], 'obs_rmse':[], 'relative_obs_rmse':[], 'latent_consistency':[], "reconstruction_diff":[], 'rec_rmse':[], 'relative_rec_rmse':[], 'lr': []}

    for step in range(n_steps):
        optimizer.zero_grad()
        x_hat = psi(model, y_hat) #psi(model, z, x_filled, quantization=quantization)           # decodes z -> x_hat (differentiable)
        # ensure x_hat has same channels as obs:
        x_hat = x_hat[..., :obs.shape[-2], :obs.shape[-1]] if x_hat.shape[-2:] != obs.shape[-2:] else x_hat
        x_hat_for_loss = x_hat
        obs_for_loss = obs
        if crop_idx is not None:
            x_hat_for_loss = x_hat_for_loss[..., crop_idx, crop_idx]
            obs_for_loss = obs_for_loss[..., crop_idx, crop_idx]
        x_hat_eval = apply_eval_postprocessing(x_hat.detach(), cfg=cfg, crop_idx=crop_idx, mlic_path=mlic_path)
        obs_eval = obs_for_loss.detach()
        # compute data term only on observed entries

        
        x_hat_eval = unorm_ssp_arr_3D(x_hat_eval.detach().cpu().numpy(), state[norm_stats + "_norm_stats"])
        obs_eval = unorm_ssp_arr_3D(obs_eval.detach().cpu().numpy(), state["test_norm_stats"])

        reconstruction_diff = x_hat_for_loss - obs_for_loss
        reconstruction_diff_eval = x_hat_eval - obs_eval
        obs_diff = mask[..., crop_idx, crop_idx] * reconstruction_diff if crop_idx is not None else mask * reconstruction_diff
        obs_diff_eval = mask[..., crop_idx, crop_idx].detach().cpu().numpy() * reconstruction_diff_eval if crop_idx is not None else mask.detach().cpu().numpy() * reconstruction_diff_eval
        # latent consistency: encode x_hat and compare to y_hat
        #with torch.no_grad():
            # to compute phi(psi(z)) we must run g_a then h_a on x_hat (we want gradient through these to z? NO)
            # But in the objective, the second term should be differentiable wrt z.
            # So we compute phi(x_hat) with gradients enabled (do NOT detach).
        y_rec = model(x_hat)['y_hat']
        latent_consistency = F.mse_loss(y_rec, y_hat)

        if loss_on == "obs":
            data_loss = obs_diff.pow(2).mean()
        elif loss_on == "full":
            data_loss = reconstruction_diff.pow(2).mean()

        loss = data_loss + lam * latent_consistency
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        torch.nn.utils.clip_grad_norm_([y_hat], max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        obs_rmse = np.sqrt(np.mean(obs_diff_eval**2))
        relative_obs_rmse = obs_rmse / n_obs if n_obs > 0 else float('inf')
        rec_rmse = np.sqrt(np.mean(reconstruction_diff_eval**2))
        relative_rec_rmse = rec_rmse / n_obs if n_obs > 0 else float('inf')

        history['loss'].append(loss.item())
        history['obs_diff'].append(np.mean(obs_diff_eval**2))
        history["obs_rmse"].append(obs_rmse)
        history['relative_obs_rmse'].append(relative_obs_rmse)
        history['latent_consistency'].append(latent_consistency.item())
        history['reconstruction_diff'].append(np.mean(reconstruction_diff_eval**2))
        history['rec_rmse'].append(rec_rmse)
        history['relative_rec_rmse'].append(relative_rec_rmse)
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_y_hat = y_hat.detach().clone()
            

        if verbose and (step % 50 == 0 or step == n_steps-1):
            print(f"[opt] step {step+1}/{n_steps} loss={loss.item():.6e} data={data_loss.item():.6e} lat={latent_consistency.item():.6e}")


        if step == 0:
            xhat_0 = x_hat.detach().clone()

    best_xhat = psi(model, best_y_hat)
    return best_y_hat, best_xhat.detach(), xhat_0, history


def compute_selected_days_rmse_summary(results, truth_tensor, state, norm_stats, cfg={}, crop_idx=None, mlic_path=None):
    """Aggregate RMSE distribution and global RMSE metrics over selected days."""

    def _distribution(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "max": None,
            }
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }

    summary = {}

    for config_key, results_per_day in results.items():
        obs_rmse_per_day = []
        rec_rmse_per_day = []
        rec_spatial_std_per_day = []

        obs_sq_sum = 0.0
        obs_count = 0
        rec_sq_sum = 0.0
        rec_count = 0

        for t_idx, result in results_per_day.items():
            best_xhat = result["best_xhat"]
            mask = result.get("mask")
            if mask is None:
                continue

            truth_sample = truth_tensor[t_idx:t_idx+1]

            x_hat_eval = apply_eval_postprocessing(best_xhat.detach(), cfg=cfg, crop_idx=crop_idx, mlic_path=mlic_path)
            truth_eval = truth_sample.detach()
            mask_eval = mask.detach()

            if crop_idx is not None:
                truth_eval = truth_eval[..., crop_idx, crop_idx]
                mask_eval = mask_eval[..., crop_idx, crop_idx]

            x_hat_eval = unorm_ssp_arr_3D(x_hat_eval.detach().cpu().numpy(), state[norm_stats + "_norm_stats"])
            truth_eval = unorm_ssp_arr_3D(truth_eval.detach().cpu().numpy(), state["test_norm_stats"])

            reconstruction_diff = x_hat_eval - truth_eval
            mask_np = mask_eval.detach().cpu().numpy().astype(bool)
            obs_diff = reconstruction_diff[mask_np]

            if obs_diff.size > 0:
                obs_rmse = float(np.sqrt(np.mean(obs_diff ** 2)))
                obs_rmse_per_day.append(obs_rmse)
                obs_sq_sum += float(np.sum(obs_diff ** 2))
                obs_count += int(obs_diff.size)

            rec_rmse = float(np.sqrt(np.mean(reconstruction_diff ** 2)))
            rec_spatial_std = float(np.std(reconstruction_diff))
            rec_rmse_per_day.append(rec_rmse)
            rec_spatial_std_per_day.append(rec_spatial_std)
            rec_sq_sum += float(np.sum(reconstruction_diff ** 2))
            rec_count += int(reconstruction_diff.size)

        summary[config_key] = {
            "n_days_evaluated": int(len(results_per_day)),
            "obs_rmse_distribution": _distribution(obs_rmse_per_day),
            "rec_rmse_distribution": _distribution(rec_rmse_per_day),
            "rec_spatial_std_distribution": _distribution(rec_spatial_std_per_day),
            "global_obs_rmse": float(np.sqrt(obs_sq_sum / obs_count)) if obs_count > 0 else None,
            "global_rec_rmse": float(np.sqrt(rec_sq_sum / rec_count)) if rec_count > 0 else None,
            "global_rec_std": float(np.sqrt(rec_sq_sum / rec_count)) if rec_count > 0 else None,
        }

    return summary


def selected_days_summary_to_dataframe(selected_days_rmse_summary, baseline_metrics=None):
    """Convert selected_days_rmse_summary into a flat DataFrame."""
    rows = []
    for config_key, summary in selected_days_rmse_summary.items():
        obs_dist = summary.get("obs_rmse_distribution", {})
        rec_dist = summary.get("rec_rmse_distribution", {})
        rec_std_dist = summary.get("rec_spatial_std_distribution", {})
        rows.append(
            {
                "config_key": config_key,
                "n_days_evaluated": summary.get("n_days_evaluated"),
                "global_obs_rmse": summary.get("global_obs_rmse"),
                "global_rec_rmse": summary.get("global_rec_rmse"),
                "global_rec_std": summary.get("global_rec_std"),
                "obs_rmse_count": obs_dist.get("count"),
                "obs_rmse_mean": obs_dist.get("mean"),
                "obs_rmse_std": obs_dist.get("std"),
                "obs_rmse_min": obs_dist.get("min"),
                "obs_rmse_p25": obs_dist.get("p25"),
                "obs_rmse_median": obs_dist.get("median"),
                "obs_rmse_p75": obs_dist.get("p75"),
                "obs_rmse_max": obs_dist.get("max"),
                "rec_rmse_count": rec_dist.get("count"),
                "rec_rmse_mean": rec_dist.get("mean"),
                "rec_rmse_std": rec_dist.get("std"),
                "rec_rmse_min": rec_dist.get("min"),
                "rec_rmse_p25": rec_dist.get("p25"),
                "rec_rmse_median": rec_dist.get("median"),
                "rec_rmse_p75": rec_dist.get("p75"),
                "rec_rmse_max": rec_dist.get("max"),
                "rec_spatial_std_count": rec_std_dist.get("count"),
                "rec_spatial_std_mean": rec_std_dist.get("mean"),
                "rec_spatial_std_std": rec_std_dist.get("std"),
                "rec_spatial_std_min": rec_std_dist.get("min"),
                "rec_spatial_std_p25": rec_std_dist.get("p25"),
                "rec_spatial_std_median": rec_std_dist.get("median"),
                "rec_spatial_std_p75": rec_std_dist.get("p75"),
                "rec_spatial_std_max": rec_std_dist.get("max"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="global_rec_rmse", ascending=True, na_position="last").reset_index(drop=True)

    if baseline_metrics is not None:
        baseline_row_dict = {col: np.nan for col in df.columns}
        baseline_row_dict["config_key"] = "baseline"

        baseline_time_rmse = np.asarray(baseline_metrics.get("time_rmse", []), dtype=np.float64)
        baseline_rec_spatial_std = np.asarray(baseline_metrics.get("rec_spatial_std_per_day", []), dtype=np.float64)

        baseline_row_dict["n_days_evaluated"] = int(baseline_time_rmse.size) if baseline_time_rmse.size > 0 else np.nan
        baseline_row_dict["global_rec_rmse"] = baseline_metrics.get("rec_rmse_mean", np.nan)

        if baseline_time_rmse.size > 0:
            baseline_row_dict["rec_rmse_count"] = int(baseline_time_rmse.size)
            baseline_row_dict["rec_rmse_mean"] = float(np.mean(baseline_time_rmse))
            baseline_row_dict["rec_rmse_std"] = float(np.std(baseline_time_rmse))
            baseline_row_dict["rec_rmse_min"] = float(np.min(baseline_time_rmse))
            baseline_row_dict["rec_rmse_p25"] = float(np.percentile(baseline_time_rmse, 25))
            baseline_row_dict["rec_rmse_median"] = float(np.percentile(baseline_time_rmse, 50))
            baseline_row_dict["rec_rmse_p75"] = float(np.percentile(baseline_time_rmse, 75))
            baseline_row_dict["rec_rmse_max"] = float(np.max(baseline_time_rmse))

        if baseline_rec_spatial_std.size > 0:
            baseline_row_dict["rec_spatial_std_count"] = int(baseline_rec_spatial_std.size)
            baseline_row_dict["rec_spatial_std_mean"] = float(np.mean(baseline_rec_spatial_std))
            baseline_row_dict["rec_spatial_std_std"] = float(np.std(baseline_rec_spatial_std))
            baseline_row_dict["rec_spatial_std_min"] = float(np.min(baseline_rec_spatial_std))
            baseline_row_dict["rec_spatial_std_p25"] = float(np.percentile(baseline_rec_spatial_std, 25))
            baseline_row_dict["rec_spatial_std_median"] = float(np.percentile(baseline_rec_spatial_std, 50))
            baseline_row_dict["rec_spatial_std_p75"] = float(np.percentile(baseline_rec_spatial_std, 75))
            baseline_row_dict["rec_spatial_std_max"] = float(np.max(baseline_rec_spatial_std))

        baseline_row = pd.DataFrame([baseline_row_dict])
        df = pd.concat([baseline_row, df], ignore_index=True)
    return df


def plot_optimization_history(reconstruction_dict,save_path=None):
   
    fig,axs = plt.subplots(2,2, figsize=(12,12))

    for results_per_day_dict in reconstruction_dict.values():

        result = next(iter(results_per_day_dict.values()))
        config = result["params"]
        n_obs = config["n_obs_per_day"]
        lam = config["lam"]
        y_hat_init = config["y_hat_init"]
        loss_on = config["loss_on"]
        k = config["k"]
        
        axs[0][0].plot(result['history']['loss'], label=f"n_obs={n_obs}, lambda={lam}, y_hat_init={y_hat_init}, loss_on={loss_on}, k={k}") 
        axs[1][0].plot(result['history']['obs_diff'], label=f"n_obs={n_obs}, lambda={lam}, y_hat_init={y_hat_init}, loss_on={loss_on}, k={k}")
        axs[0][1].plot(result['history']['latent_consistency'], label=f"n_obs={n_obs}, lambda={lam}, y_hat_init={y_hat_init}, loss_on={loss_on}, k={k}")
        axs[1][1].plot(result['history']['reconstruction_diff'], label=f"n_obs={n_obs}, lambda={lam}, y_hat_init={y_hat_init}, loss_on={loss_on}, k={k}")

    time = config.get("time_value", None)

    axs[0][0].set_title("Total Loss")
    axs[1][0].set_title("observation mse")
    axs[0][1].set_title("Latent Consistency Loss")
    axs[1][1].set_title("Reconstruction mse")

    for ax_row in axs:
        for ax in ax_row:   
            ax.set_xlabel("Optimization Step")
            ax.legend(fontsize=8)

    fig.suptitle(f"Optimization History at time={time}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(f"{save_path}/optimization_history.png", dpi=300)


def plot_profiles(reconstruction_dict, ssp_ae_tensor, truth_tensor, mask, depth_array, save_path=None):

    fig,ax = plt.subplots(1,1, figsize=(8,6))

    profiles_idx = ((mask[:,0,:,:] == 1).nonzero(as_tuple=False))

    if len(profiles_idx) == 0:
        print("No observed profiles found in the mask. Skipping profile plot.")
        return

    idx = profiles_idx[0].detach().cpu().numpy()
    lat,lon = idx[1], idx[2]

    for results_per_day_dict in reconstruction_dict.values():


        result = next(iter(results_per_day_dict.values()))
        config = result["params"]
        n_obs = config["n_obs_per_day"]
        lam = config["lam"]
        y_hat_init = config["y_hat_init"]
        loss_on = config["loss_on"]
        k = config["k"]
        t_idx = config.get("t_idx", None)
        t = config.get("time_value", None)


        ax.plot(result['best_xhat'][t_idx,:,lat,lon].detach().cpu().numpy(), depth_array, label=f"n_obs={n_obs}, lambda={lam}, y_hat_init={y_hat_init}, k={k}, loss_on={loss_on}") 


    ax.plot(truth_tensor[t_idx,:,lat,lon].detach().cpu().numpy(), depth_array, label=f"obs")
    ax.plot(ssp_ae_tensor[t_idx,:,lat,lon].detach().cpu().numpy(), depth_array, label=f"baseline reconstruction")
            

    ax.set_xlabel("Value")
    ax.set_ylabel("Depth")
    ax.legend()
    ax.invert_yaxis()
    fig.suptitle(f"Profiles at t={t}, lat_idx={lat}, lon_idx={lon}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(f"{save_path}/optimized_profiles.png", dpi=300)


def plot_first_day_error_map(
    reconstruction_dict,
    truth_tensor,
    state,
    norm_stats,
    cfg = {},
    crop_idx=None,
    mlic_path=None,
    config_key=None,
    day_idx=None,
    save_path=None,
):
    """Plot first optimized day spatial RMSE map with observed points as crosses."""
    if not reconstruction_dict:
        raise ValueError("reconstruction_dict is empty")

    if config_key is None:
        config_key = next(iter(reconstruction_dict.keys()))
    if config_key not in reconstruction_dict:
        raise ValueError(f"config_key '{config_key}' not found in reconstruction_dict")

    results_per_day = reconstruction_dict[config_key]
    if not results_per_day:
        raise ValueError(f"No day results found for config_key '{config_key}'")

    day_keys = sorted(results_per_day.keys())
    if day_idx is None:
        day_idx = int(day_keys[0])
    if day_idx not in results_per_day:
        raise ValueError(f"day_idx={day_idx} not found for config_key '{config_key}'")

    result = results_per_day[day_idx]
    best_xhat = result["best_xhat"].detach()
    mask = result.get("mask")
    if mask is None:
        raise ValueError("mask is required in result payload to plot observation points")

    truth_sample = truth_tensor[day_idx : day_idx + 1].detach()

    x_hat_eval = apply_eval_postprocessing(best_xhat, cfg=cfg, crop_idx=crop_idx, mlic_path=mlic_path)
    truth_eval = truth_sample
    mask_eval = mask.detach()

    if crop_idx is not None:
        truth_eval = truth_eval[..., crop_idx, crop_idx]
        mask_eval = mask_eval[..., crop_idx, crop_idx]

    x_hat_eval = unorm_ssp_arr_3D(x_hat_eval.detach().cpu().numpy(), state[norm_stats + "_norm_stats"])
    truth_eval = unorm_ssp_arr_3D(truth_eval.detach().cpu().numpy(), state["test_norm_stats"])

    diff_unorm = x_hat_eval - truth_eval
    # Spatial map by aggregating depth-wise error at each (lat, lon).
    error_map = np.sqrt(np.mean(diff_unorm[0] ** 2, axis=0))

    obs_points = mask_eval[0, 0].detach().cpu().numpy().astype(bool)
    yy, xx = np.where(obs_points)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    im = ax.imshow(error_map, origin="lower", cmap="magma")
    ax.scatter(xx, yy, marker="x", s=30, c="cyan", linewidths=0.8, label="observations")

    time_value = result.get("params", {}).get("time_value", str(day_idx))
    ax.set_title(f"First-day reconstruction error map (RMSE over depth) | t={time_value}")
    ax.set_xlabel("Lon index")
    ax.set_ylabel("Lat index")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(f"{save_path}/first_day_error_with_obs_crosses.png", dpi=300)


if __name__ == "__main__":
    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = os.environ.get("FSC_DEVICE", preferred_device)
    load_device = "cpu"
    verbose = True
    crop_idx = slice(20, -20)
    time_sample=1
    norm_stats = "test"
    torch.manual_seed(42)

    mlic_path = "/Odyssey/private/o23gauvr/code/MLIC/experiments/test_mean_std_along_depth/fixed_weight_loss_64_96_1.0_CR_10000.0_enatl_natl__mean_std_along_depth/20260726_101454/checkpoints/best_checkpoint_rmse.pth.tar" #"/Odyssey/private/o23gauvr/code/MLIC/experiments/test_mix/fixed_weight_loss_64_96_1.0_CR_10000.0_enatl_natl__mean_std/20260723_220415/checkpoints/best_checkpoint_rmse.pth.tar" #"/Odyssey/private/o23gauvr/code/MLIC/experiments/article_long_new/filtered_z_uniform_dm_alternate_days_classic_loss/fixed_weight_loss_64_96_1.0_CR_10000.0_enatl_natl__mean_std/20260619_180735/checkpoints/best_checkpoint_rmse.pth.tar" 
    dm_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split_filtered_z_uniform_alternate_days_7_60_10.pkl"

    with open(dm_path, "rb") as f:
        dm = pickle.load(f)

    model, state, model_type, cfg = load_model_from_checkpoint(str(mlic_path), device=load_device, dm=dm, batch=None)

    if device != "cpu":
        try:
            model = model.to(device)
        except torch.OutOfMemoryError:
            print("CUDA OOM while moving model to GPU; falling back to CPU execution.")
            device = "cpu"
            model = model.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    truth_da = dm.test_ds.input
    time_values = truth_da.time.values

    # Keep training data on CPU; only active mini-batches are moved to CUDA when needed.
    x_train = torch.tensor(dm.train_ds.input.data.copy(), dtype=torch.float32)



    ssp_truth_da = dm.test_ds.input
    ssp_truth_da_unorm = unorm_ssp_arr_3D(ssp_truth_da.values, ssp_truth_da.attrs['norm_stats'])
    ssp_truth_unorm = ssp_truth_da_unorm[::time_sample]  # Subsample for faster metric computation


    depth_array = ssp_truth_da.z.values
    # dm_test_norm = ssp_truth_da.attrs['norm_stats']
    # dm_train_norm = dm.train_ds.input.attrs['norm_stats']

    infer_batch_size = int(os.environ.get("MLIC_INFER_BATCH_SIZE", "16"))
    truth_tensor, ssp_ae_tensor = load_mlic(
        ssp_truth_unorm,
        state,
        norm_stats,
        mlic_path,
        model,
        cfg=cfg,
        device=device,
        batch_size=infer_batch_size,
    )

    # Keep full-series tensors on CPU and move per-sample tensors to CUDA inside the loop.
    truth_tensor = truth_tensor.cpu()
    ssp_ae_tensor = ssp_ae_tensor.cpu()


    B, C, H, W = truth_tensor.shape


    # ssp_truth_da = ssp_truth_da.isel(lat=crop_idx, lon=crop_idx)
    # ssp_ae_da = ssp_ae_da.isel(lat=crop_idx, lon=crop_idx)

    
    #ssp_ae_tensor = torch.from_numpy(ssp_ae_da.values.astype(ssp_ae_da.dtype)).to(device)

    del ssp_truth_da, ssp_truth_da_unorm, ssp_truth_unorm #, ssp_ae_da



    ###########################################################################################################################################################



    lr = 1e-2
    n_steps = 2
    cadence_results = {}
    selected_indices_by_cadence = {}
    per_cadence_details = {}
    max_obs = int(x_train.shape[2]*x_train.shape[3])
    n_obs_per_day_values = [100] #0, 100, 1000, 10000, max_obs
    y_hat_inits = [ "nn"] # , "0_filled", "obs_mean_profile","train_mean_profile", "spline_interp","train_mean_profile", "spline_interp",
    loss_on_values = ["full"] #
    lam_values = [0.0] #0.0, 0.5, 1.0, 10.0
    max_available_k = max(1, int(x_train.shape[0]))
    nn_k_values = [80] #1, 4, 10, 100, max_available_k
    n_days = 1
    days_idx_top_optimise = np.linspace(0, len(time_values), n_days, dtype=int, endpoint=False)
    sensor_mode = os.environ.get("SENSOR_PLACEMENT_MODE", "latent_space_qr").strip().lower()  #qr_pivot, learned_soft, baseline
    sensor_qr_rank = int(os.environ.get("SENSOR_QR_RANK", "10"))
    sensor_mask_outer_steps = int(os.environ.get("SENSOR_MASK_OUTER_STEPS", "25"))
    sensor_mask_inner_steps = int(os.environ.get("SENSOR_MASK_INNER_STEPS", "25"))
    sensor_mask_budget_lambda = float(os.environ.get("SENSOR_BUDGET_LAMBDA", "1.0"))
    sensor_mask_entropy_lambda = float(os.environ.get("SENSOR_ENTROPY_LAMBDA", "1e-3"))

    qr_candidate_indices = None
    if sensor_mode == "qr_pivot":
        qr_candidate_indices = qr_pivot_sensor_candidates(
            dm.train_ds.input.values,
            budget=max_obs,
            rank=sensor_qr_rank,
        )

    elif sensor_mode == "latent_space_qr":
        score = compute_sensor_score(model, x_train, crop_idx=crop_idx, batch_size=infer_batch_size, device=device)
        #score = compute_sensor_score_fast(model, x_train[::4], crop_idx=slice(20,-20),n_probe=100, batch_size=infer_batch_size, device=device)



    results = {}


    baseline_metrics = evaluate_full_span_reconstruction(
        ssp_ae_tensor[days_idx_top_optimise],
        truth_tensor[days_idx_top_optimise],
        state,
        cfg=cfg,
        norm_stats=norm_stats,
        crop_idx=crop_idx,
        mlic_path=mlic_path,
    )

    for y_hat_init in y_hat_inits:
        k_values = nn_k_values if y_hat_init == "nn" else [None]
        for k in tqdm(k_values, desc=f"y_hat_init={y_hat_init}"):
            for loss_on in loss_on_values:
                for lam in lam_values:
                    for n_obs_per_day in n_obs_per_day_values:

                        config_key = (
                            f"mode={sensor_mode}|nobs={n_obs_per_day}|init={y_hat_init}|"
                            f"k={k if k is not None else 'na'}|loss={loss_on}|lam={lam}"
                            )

                        results.setdefault(config_key, {})

                        for t_idx in days_idx_top_optimise:

                            obs_sample = truth_tensor[t_idx : t_idx + 1].to(device)  # shape [1, C, H, W]

                            if sensor_mode == "learned_soft":
                                best_y_hat, best_xhat, xhat_0, learned_mask, history = learn_sensor_mask_with_latent_optimization(
                                    model,
                                    obs_sample,
                                    state,
                                    cfg=cfg,
                                    loss_on=loss_on,
                                    x_train=x_train,
                                    y_hat_init=y_hat_init,
                                    k=k,
                                    nn_encode_batch_size=infer_batch_size,
                                    crop_idx=crop_idx,
                                    mlic_path=mlic_path,
                                    n_outer_steps=sensor_mask_outer_steps,
                                    y_steps_per_outer=sensor_mask_inner_steps,
                                    mask_lr=lr,
                                    y_lr=lr,
                                    lam=lam,
                                    lam_budget=sensor_mask_budget_lambda,
                                    lam_entropy=sensor_mask_entropy_lambda,
                                    budget=n_obs_per_day,
                                    device=device,
                                    verbose=verbose,
                                )
                                mask = learned_mask.unsqueeze(1).expand_as(obs_sample) if learned_mask is not None else make_spatial_observation_mask(obs_sample, n_obs=n_obs_per_day, seed=42)
                            else:
                                if sensor_mode == "qr_pivot":
                                    candidate_count = max(0, min(int(n_obs_per_day), len(qr_candidate_indices))) if qr_candidate_indices is not None else 0
                                    candidate_indices = qr_candidate_indices[:candidate_count] if qr_candidate_indices is not None else []
                                    mask = build_mask_from_flat_indices(obs_sample, candidate_indices)
                                    
                                #)
                                elif sensor_mode == "latent_space_qr":
                                    candidate_indices = (
                                        score.flatten()
                                        .argsort(descending=True)[:n_obs_per_day]
                                        .cpu()
                                        .tolist()
                                        )
                                    mask = build_mask_from_flat_indices(obs_sample, candidate_indices)

                                else:
                                    mask = make_spatial_observation_mask(obs_sample, n_obs=n_obs_per_day, seed=42) #+ int(t_idx)

                                best_y_hat, best_xhat, xhat_0, history = optimize_latent_from_observations(
                                    model,
                                    obs_sample,
                                    mask,
                                    state,
                                    cfg=cfg,
                                    loss_on=loss_on,
                                    x_train=x_train,
                                    y_hat_init=y_hat_init,
                                    k=k,
                                    nn_encode_batch_size=infer_batch_size,
                                    crop_idx=crop_idx,
                                    mlic_path=mlic_path,
                                    n_steps=n_steps,
                                    lr=lr,
                                    lam=lam,
                                    device=device,
                                    verbose=verbose,
                                )

                            results.setdefault(config_key, {}).setdefault(int(t_idx), {})["best_xhat"] = best_xhat.detach().clone()
                            results[config_key][int(t_idx)]["best_y_hat"] = best_y_hat.detach().clone()
                            results[config_key][int(t_idx)]["xhat_0"] = xhat_0.detach().clone()
                            results[config_key][int(t_idx)]["history"] = history
                            results[config_key][int(t_idx)]["mask"] = mask.detach().clone()
                            results[config_key][int(t_idx)]["params"] = {
                                "n_obs_per_day": n_obs_per_day,
                                "y_hat_init": y_hat_init,
                                "k": k,
                                "loss_on": loss_on,
                                "lam": lam,
                                "t_idx": t_idx,
                                "time_value": time_values[t_idx].astype("datetime64[D]").astype(str),
                            }

    selected_days_rmse_summary = compute_selected_days_rmse_summary(
        results,
        truth_tensor,
        state,
        cfg=cfg,
        norm_stats=norm_stats,
        crop_idx=crop_idx,
        mlic_path=mlic_path,
    )

    selected_days_rmse_df = selected_days_summary_to_dataframe(selected_days_rmse_summary, baseline_metrics)


    results_dir = Path("/Odyssey/private/o23gauvr/code/FASCINATION")
    pickle_path = results_dir / "pickle" / "z_optimization_cadence_results_test_2.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(
            {
                "baseline_metrics": baseline_metrics,
                "results": results,
                "selected_days_rmse_summary": selected_days_rmse_summary,
            },
            f,
        )

    for config_key, summary in selected_days_rmse_summary.items():

        print(
            f"{config_key} | global_rec_rmse={summary['global_rec_rmse']:.6e} | "
            f"{config_key} | rec_rmse_std={summary['rec_rmse_distribution']['std']:.6e}"
            )
        
        if summary['global_obs_rmse'] is None or summary['global_rec_rmse'] is None:
            print(f"{config_key} | global_obs_rmse=None | obs_rmse_std=None | global_rec_rmse=None | rec_rmse_std=None")
            continue
        print(
            f"{config_key} | global_obs_rmse={summary['global_obs_rmse']:.6e} | "
            f"{config_key} | obs_rmse_std={summary['obs_rmse_distribution']['std']:.6e}"
        )



    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_results_dir = Path(f"/Odyssey/private/o23gauvr/code/FASCINATION/experiments/latent_space_optimisation/results_test_full_loss_gauss/{date}")
    summary_results_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "crop_idx": {
            "start": crop_idx.start,
            "stop": crop_idx.stop,
            "step": crop_idx.step,
        },
        "time_sample": time_sample,
        "norm_stats": norm_stats,
        "torch_seed": 42,
        "mlic_path": mlic_path,
        "dm_path": dm_path,
        "n_steps": n_steps,
        "cadence_results": cadence_results,
        "selected_indices_by_cadence": selected_indices_by_cadence,
        "per_cadence_details": per_cadence_details,
        "n_obs_per_day_values": n_obs_per_day_values,
        "y_hat_inits": y_hat_inits,
        "loss_on_values": loss_on_values,
        "lam_values": lam_values,
        "max_available_k": max_available_k,
        "nn_k_values": nn_k_values,
        "n_days": n_days,
        "lr": lr,
        "infer_batch_size": infer_batch_size,
        "device": device,
        "load_device": load_device,
        "model_type": model_type,
    }
    with open(summary_results_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    selected_days_rmse_df.to_csv(summary_results_dir / "selected_days_rmse_summary_sorted.csv", index=False)
    selected_days_rmse_df.to_pickle(summary_results_dir / "selected_days_rmse_summary_sorted.pkl")

    plot_optimization_history(results, save_path=summary_results_dir)
    plot_profiles(results, ssp_ae_tensor, truth_tensor, mask, depth_array, save_path=summary_results_dir)
    plot_first_day_error_map(
        results,
        truth_tensor,
        state,
        cfg=cfg,
        norm_stats=norm_stats,
        crop_idx=crop_idx,
        mlic_path=mlic_path,
        save_path=summary_results_dir,
    )




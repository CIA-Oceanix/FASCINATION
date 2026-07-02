#!/usr/bin/env python3
"""
Example script demonstrating RAMGEO metrics computation in test_metrics.py

This script shows how to:
1. Load compression models (MLIC, CAE)
2. Select random slices with a seed
3. Run RAMGEO simulations
4. Compute transmission loss metrics (MAE, MS-SSIM)
5. Export results to CSV
"""

import sys
import os
sys.path.insert(0, "/Odyssey/private/o23gauvr/code")
sys.path.insert(0, "/Odyssey/private/o23gauvr/code/FASCINATION")

import pickle
import torch
import numpy as np
from pathlib import Path
from FASCINATION.src.test_metrics import (
    run_ramgeo_and_compute_metrics,
    select_random_slices,
    process_model_in_batches,
)
from FASCINATION.src.utils import unorm_ssp_arr_3D, get_cfg_from_ckpt_path, load_model, norm_ssp_arr_3D
from MLIC.MLIC.models import MLICPlusPlus
from MLIC.MLIC.utils.utils import Config
import torch.nn as nn


def main():
    """Main workflow for RAMGEO metrics computation."""
    
    print("="*80)
    print("RAMGEO Metrics Computation Example")
    print("="*80)
    
    # ========== LOAD DATA ==========
    print("\n[1] Loading data...")
    dm_data_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_157_196_256_norm_per_split.pkl"
    with open(dm_data_path, 'rb') as f:
        dm = pickle.load(f)
    
    ssp_da = dm.test_ds.input
    depth_array = ssp_da.z.values
    test_norm = ssp_da.attrs['norm_stats']
    season_idx = dm.test_ds.input.season_idx
    sst_test_norm = dm.test_ds.input.attrs['sst'].data
    
    ssp_truth = unorm_ssp_arr_3D(ssp_da.values, test_norm)
    print(f"✓ Data loaded. SSP truth shape: {ssp_truth.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # ========== LOAD MLIC ==========
    print("\n[2] Loading MLIC model...")
    mlic_ckpt_path = Path("/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/eusipco/MLIC/SSP/fixed_weight_loss_64_96_1.0_CR_1000.0_seed_42/20250917_151130/checkpoints/checkpoint_best_loss.pth.tar")
    
    if not mlic_ckpt_path.exists():
        print(f"✗ MLIC checkpoint not found: {mlic_ckpt_path}")
        print("  Please update the path to your MLIC checkpoint")
        return
    
    def parse_experiment_config(ckpt_path):
        """Parse experiment config from checkpoint."""
        config_file = list((ckpt_path.parent.parent).rglob("train_*.log"))
        if not config_file:
            return None
        with open(config_file[0], 'r') as f:
            content = f.read()
            lines = content.split('\n')
            cfg = eval((lines[1].strip().split("INFO: ")[-1]).replace("<class 'torch.nn.modules.activation.","nn.").replace("'>",""))
        return cfg
    
    cfg = parse_experiment_config(mlic_ckpt_path)
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
    ck = torch.load(mlic_ckpt_path, map_location=device)
    net.load_state_dict(ck['state_dict'])
    test_norm_mlic = ck.get('train_norm_stats', None)
    
    # Normalize truth for MLIC
    ssp_truth_mlic = ssp_truth.copy()
    if test_norm_mlic['method'] == "min_max":
        x_min = test_norm_mlic["params"]["x_min"].astype(ssp_truth_mlic.dtype)
        x_max = test_norm_mlic["params"]["x_max"].astype(ssp_truth_mlic.dtype)
        ssp_truth_mlic = (ssp_truth_mlic - x_min) / (x_max - x_min)
    
    ssp_truth_tens = torch.tensor(ssp_truth_mlic).to(device=device, dtype=getattr(torch, dm.dtype_str))
    with torch.no_grad():
        rv_batch = net(ssp_truth_tens, season_idx, sst_test_norm)
    
    ssp_ae_mlic = rv_batch['x_hat'].detach().cpu().numpy()
    ssp_ae_mlic = unorm_ssp_arr_3D(ssp_ae_mlic, test_norm_mlic)
    ssp_ae_mlic_da = ssp_da.copy(data=ssp_ae_mlic)
    print(f"✓ MLIC loaded. Output shape: {ssp_ae_mlic_da.shape}")
    
    # ========== LOAD CAE ==========
    print("\n[3] Loading CAE model...")
    cae_ckpt_path = Path("/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/eusipco/CAE/SSP/baseline_cae_32_64_CR_1000_seed_42/20250917_154130/checkpoints/checkpoint_best_rmse.pth.tar")
    
    if not cae_ckpt_path.exists():
        print(f"✗ CAE checkpoint not found: {cae_ckpt_path}")
        print("  Please update the path to your CAE checkpoint")
        return
    
    cfg = get_cfg_from_ckpt_path(str(cae_ckpt_path), pprint=False)
    ck = torch.load(cae_ckpt_path, map_location=device)
    test_norm_cae = ck.get('norm_stats', None)
    
    ssp_truth_cae = unorm_ssp_arr_3D(ssp_da.values, test_norm)
    if test_norm_cae['method'] == "min_max":
        x_min = test_norm_cae["params"]["x_min"].astype(ssp_truth_cae.dtype)
        x_max = test_norm_cae["params"]["x_max"].astype(ssp_truth_cae.dtype)
        ssp_truth_cae = (ssp_truth_cae - x_min) / (x_max - x_min)
    
    ssp_truth_tens = torch.tensor(ssp_truth_cae).to(device=device, dtype=getattr(torch, dm.dtype_str))
    lit_model = load_model(str(cae_ckpt_path), dm, ssp_truth_tens, verbose=False)
    
    batch_size = 4
    print(f"Processing {ssp_truth_tens.shape[0]} samples in batches of {batch_size}...")
    ssp_ae_cae = process_model_in_batches(lit_model, ssp_truth_tens, batch_size=batch_size, dim=0, device=device)
    
    ssp_ae_cae = unorm_ssp_arr_3D(ssp_ae_cae, test_norm_cae)
    ssp_ae_cae_da = ssp_da.copy(data=ssp_ae_cae)
    print(f"✓ CAE loaded. Output shape: {ssp_ae_cae_da.shape}")
    
    # ========== SELECT RANDOM SLICES ==========
    print("\n[4] Selecting random slices...")
    n_slices = 5  # Select 5 random slices
    random_seed = 42
    selected_slices = select_random_slices(ssp_truth.shape, n_slices=n_slices, random_seed=random_seed)
    print(f"✓ Selected {len(selected_slices)} slices (with seed={random_seed}):")
    for i, (t, lat) in enumerate(selected_slices):
        print(f"  [{i+1}] time_idx={t}, lat_idx={lat}")
    
    # ========== RUN RAMGEO AND COMPUTE METRICS ==========
    print("\n[5] Running RAMGEO simulations and computing metrics...")
    output_dir = "/Odyssey/private/o23gauvr/code/RAMGEO2025/RAMGEO2025/data/test_metrics_ramgeo_example"
    
    freq_list = [10, 100, 500, 1000]
    
    df_results = run_ramgeo_and_compute_metrics(
        ssp_truth_da=ssp_da,
        ssp_ae_mlic_da=ssp_ae_mlic_da,
        ssp_ae_cae_da=ssp_ae_cae_da,
        selected_slices=selected_slices,
        freq_list=freq_list,
        output_dir=output_dir,
        random_seed=random_seed,
        device=device,
    )
    
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    if len(df_results) > 0:
        print("\nDetailed Results (first 10 rows):")
        print(df_results.head(10).to_string())
        print(f"\nTotal rows: {len(df_results)}")
        print(f"Output directory: {output_dir}")
    
    return df_results


if __name__ == "__main__":
    df = main()

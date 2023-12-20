import numpy as np
import xrft
import torch
import xarray as xr
import matplotlib.pyplot as plt

np.random.seed(42) # setting seed for random permutation

def half_lr_adam(lit_mod, lr):
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ],
    )


def cosanneal_lr_adam(lit_mod, lr, T_max=100):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ],
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }


def triang_lr_adam(lit_mod, lr_min=5e-5, lr_max=3e-3, nsteps=200):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_max},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_max / 2},
        ],
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=nsteps,
            step_size_down=nsteps,
            gamma=0.95,
            cycle_momentum=False,
            mode="exp_range",
        ),
    }

def load_sound_speed_fields(path):
    ssf = xr.open_dataset(path).transpose("time", "z", "lat", "lon")
    shuffled_index = np.random.permutation(len(ssf.time))

    return ssf.isel(time=shuffled_index) # shuffling data to better acount for seasonal variability

def load_ssf_acoustic_variables(path1, path2):
    ssf = xr.open_dataset(path1).transpose("time", "lat", "lon", "z")
    cutoff_ecs = xr.open_dataset(path2).transpose("time", "lat", "lon")
    
    for var in cutoff_ecs.data_vars:
        cutoff_ecs[var] = xr.where(cutoff_ecs[var] == 999999999999.0, 0, cutoff_ecs[var]) # setting infinite values to 0
    cutoff_ecs["cutoff_freq"] = xr.where(cutoff_ecs["cutoff_freq"] > 10000, 10000, cutoff_ecs["cutoff_freq"]) # capping cutoff frequency at 10kHz to make training more stable. We are also only interested about frequencies around 1kHz

    return ssf, cutoff_ecs

def rmse_based_scores(da_rec, da_ref):
    rmse_t = (
        1.0
        - (((da_rec - da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
        / (((da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
    )
    rmse_xy = (((da_rec - da_ref) ** 2).mean(dim=("time"))) ** 0.5
    rmse_t = rmse_t.rename("rmse_t")
    rmse_xy = rmse_xy.rename("rmse_xy")
    reconstruction_error_stability_metric = rmse_t.std().values
    leaderboard_rmse = (
        1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / (((da_ref) ** 2).mean()) ** 0.5
    )
    return (
        rmse_t,
        rmse_xy,
        np.round(leaderboard_rmse.values, 5),
        np.round(reconstruction_error_stability_metric, 5),
    )


def psd_based_scores(da_rec, da_ref):
    err = da_rec - da_ref
    err["time"] = (err.time - err.time[0]) / np.timedelta64(1, "D")
    signal = da_ref
    signal["time"] = (signal.time - signal.time[0]) / np.timedelta64(1, "D")
    psd_err = xrft.power_spectrum(
        err, dim=["lat", "lon"], detrend="constant", window="hann"
    ).compute()
    psd_signal = xrft.power_spectrum(
        signal, dim=["lat", "lon"], detrend="constant", window="hann"
    ).compute()
    mean_psd_signal = psd_signal.mean(dim="time").where(
        (psd_signal.freq_lon > 0.0) & (psd_signal.freq_lat > 0.0), drop=True
    )
    mean_psd_err = psd_err.mean(dim="time").where(
        (psd_err.freq_lon > 0.0) & (psd_err.freq_lat > 0.0), drop=True
    )
    psd_based_score = 1.0 - mean_psd_err / mean_psd_signal
    level = [0.5]
    cs = plt.contour(
        1.0 / psd_based_score.freq_lon.values,
        1.0 / psd_based_score.freq_lat.values,
        psd_based_score,
        level,
    )
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    psd_da = 1.0 - mean_psd_err / mean_psd_signal
    psd_da.name = "psd_score"
    return (
        psd_da.to_dataset(),
        np.round(shortest_spatial_wavelength_resolved, 3)
    )
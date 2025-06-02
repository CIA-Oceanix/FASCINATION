import sys
import os

running_path = "/Odyssey/private/o23gauvr/code/"
sys.path.insert(0,running_path)
os.chdir(running_path)


import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from sklearn.decomposition import PCA
import torch

from FASCINATION.src.utils import get_min_max_idx, get_f1_score, cubic_interpolate_along_axis, compute_psnr, compute_msssim

def main():
    data_path = {
        "enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
        "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"
    }

    da_natl = xr.open_dataarray(data_path["natl"])
    da_enatl = xr.open_dataarray(data_path["enatl"])

    da_natl = da_natl.sel(z=da_natl.z[da_natl.z < 2000])
    da_enatl = da_enatl.sel(z=da_enatl.z[da_enatl.z < 2000])

    da_natl = da_natl.dropna(dim="lat")
    da_enatl = da_enatl.dropna(dim="lat")

    def assign_season(ds):
        month = ds['time.month']
        return xr.where(
            (month == 12) | (month <= 2), "winter",
            xr.where((month >= 3) & (month <= 5), "spring",
            xr.where((month >= 6) & (month <= 8), "summer", "autumn"))
        )

    da_natl = da_natl.assign_coords(season=assign_season(da_natl))
    da_enatl = da_enatl.assign_coords(season=assign_season(da_enatl))

    common_z = np.intersect1d(da_natl.z.values, da_enatl.z.values)
    common_lat = np.intersect1d(da_natl.lat.values, da_enatl.lat.values)
    common_lon = np.intersect1d(da_natl.lon.values, da_enatl.lon.values)

    da_natl = da_natl.sel(z=common_z, lat=common_lat, lon=common_lon)
    da_enatl = da_enatl.sel(z=common_z, lat=common_lat, lon=common_lon)

    explained_variance_thresholds = [0.5, 0.8, 0.9, 0.95, 0.98]
    season_metrics = {}

    # Add "all" to the list of seasons
    for season in tqdm(["winter", "spring", "summer", "autumn", "all"], desc="Processing seasons", unit="season"):
        print(f"\nProcessing season: {season}")

        if season == "all":
            # Use all available time points, matching natl and enatl
            da_natl_season = da_natl.dropna(dim="time")
            da_enatl_season = da_enatl.dropna(dim="time")
        else:
            da_natl_season = da_natl.sel(time=da_natl['season'] == season).dropna(dim="time")
            da_enatl_season = da_enatl.sel(time=da_enatl['season'] == season).dropna(dim="time")

        # Ensure both have the same shape (time, z, lat, lon)
        min_time = min(da_natl_season.sizes["time"], da_enatl_season.sizes["time"])
        da_natl_season = da_natl_season.isel(time=slice(0, min_time))
        da_enatl_season = da_enatl_season.isel(time=slice(0, min_time))

        natl_arr = da_natl_season.values
        enatl_arr = da_enatl_season.values

        t, z, lat, lon = natl_arr.shape
        season_metrics[season] = {}

        for threshold in tqdm(explained_variance_thresholds, desc="Processing thresholds", unit="threshold"):
            # Depth PCA
            depth_pca = PCA()
            depth_pca.fit(natl_arr.transpose(0, 2, 3, 1).reshape(-1, z))
            cumulative_variance_depth = np.cumsum(depth_pca.explained_variance_ratio_)
            n = np.argmax(cumulative_variance_depth >= threshold) + 1

            depth_pca = PCA(n_components=n)
            natl_depth_pca = depth_pca.fit_transform(natl_arr.transpose(0, 2, 3, 1).reshape(-1, z))
            enatl_depth_pca = depth_pca.transform(enatl_arr.transpose(0, 2, 3, 1).reshape(-1, z))

            natl_arr_reduced_depth = natl_depth_pca.reshape(t, lat, lon, n).transpose(0, 3, 1, 2)
            enatl_arr_reduced_depth = enatl_depth_pca.reshape(t, lat, lon, n).transpose(0, 3, 1, 2)

            # Spatial PCA
            spatial_pca = PCA()
            spatial_pca.fit(natl_arr_reduced_depth.reshape(-1, lat * lon))
            cumulative_variance_spatial = np.cumsum(spatial_pca.explained_variance_ratio_)
            m = np.argmax(cumulative_variance_spatial >= threshold) + 1

            spatial_pca = PCA(n_components=m)
            natl_spatial_pca = spatial_pca.fit_transform(natl_arr_reduced_depth.reshape(-1, lat * lon))
            enatl_spatial_pca = spatial_pca.transform(enatl_arr_reduced_depth.reshape(-1, lat * lon))

            natl_arr_reduced_spatial = natl_spatial_pca.reshape(t, n, m)
            enatl_arr_reduced_spatial = enatl_spatial_pca.reshape(t, n, m)

            # Time PCA
            natl_arr_time = natl_arr_reduced_spatial.transpose(1, 2, 0).reshape(n * m, t).T
            enatl_arr_time = enatl_arr_reduced_spatial.transpose(1, 2, 0).reshape(n * m, t).T

            time_pca = PCA()
            time_pca.fit(natl_arr_time)
            cumulative_variance_time = np.cumsum(time_pca.explained_variance_ratio_)
            k = np.argmax(cumulative_variance_time >= threshold) + 1

            time_pca = PCA(n_components=k)
            natl_time_pca = time_pca.fit_transform(natl_arr_time)
            enatl_time_pca = time_pca.transform(enatl_arr_time)

            # Inverse transform for reconstruction (on eNATL)
            enatl_time_recon = time_pca.inverse_transform(enatl_time_pca)
            enatl_time_recon = enatl_time_recon.T.reshape(n, m, t).transpose(2, 0, 1)

            enatl_spatial_recon = spatial_pca.inverse_transform(enatl_time_recon.reshape(-1, m))
            enatl_spatial_recon = enatl_spatial_recon.reshape(t, n, lat, lon).transpose(0, 2, 3, 1)

            enatl_depth_recon = depth_pca.inverse_transform(enatl_spatial_recon.reshape(-1, n))
            enatl_depth_recon = enatl_depth_recon.reshape(t, lat, lon, z).transpose(0, 3, 1, 2)

            arr_truth = enatl_arr
            arr_pred = enatl_depth_recon

            original_size = arr_truth.nbytes
            reduced_size = enatl_time_pca.nbytes
            cr = original_size / reduced_size

            max_ssp_truth_idx = np.nanargmax(arr_truth, axis=1)
            ecs_truth = da_enatl_season.z.values[max_ssp_truth_idx]
            max_ssp_pred_idx = np.nanargmax(arr_pred, axis=1)
            ecs_pred = da_enatl_season.z.values[max_ssp_pred_idx]

            psnr = compute_psnr(arr_truth, arr_pred)
            msssim = compute_msssim(
                torch.tensor(cubic_interpolate_along_axis(arr_truth, 161, axis=2), dtype=torch.float64),
                torch.tensor(cubic_interpolate_along_axis(arr_pred, 161, axis=2), dtype=torch.float64),
            )
            ae_ssp_rmse = np.sqrt(np.mean((arr_truth - arr_pred) ** 2))
            ae_ecs_rmse = np.sqrt(np.mean((ecs_truth - ecs_pred) ** 2))
            mae = np.mean(np.abs(arr_truth - arr_pred))

            min_max_idx_truth = get_min_max_idx(arr_truth, pad=False)
            min_max_idx_pred = get_min_max_idx(arr_pred, pad=False)
            mean_error_n_min_max = np.mean(np.abs(np.sum(min_max_idx_truth, axis=1) - np.sum(min_max_idx_pred, axis=1)))

            F1_score = get_f1_score(min_max_idx_truth, min_max_idx_pred)
            f1_score = np.mean(F1_score)

            r2 = 1 - (np.sum((arr_truth - arr_pred) ** 2) /
                      np.sum((arr_truth - np.mean(arr_truth)) ** 2))

            season_metrics[season][threshold] = {
                "CR": cr,
                "RMSE": ae_ssp_rmse,
                "PSNR": psnr,
                "MS-SSIM": msssim,
                "ECS": ae_ecs_rmse,
                "MAE": mae,
                "mean_error_n_min_max": mean_error_n_min_max,
                "F1_score": f1_score,
                "R2_score": r2,
                "n_depth": n,
                "m_spatial": m,
                "k_time": k
            }

    season_metrics_df = {season: pd.DataFrame.from_dict(season_metrics[season], orient='index') for season in season_metrics}
    for season, df in season_metrics_df.items():
        print(f"\nMetrics for {season}:")
        print(df)

    # Save all metrics to CSV
    for season, df in season_metrics_df.items():
        df.to_csv(f"/Odyssey/private/o23gauvr/code/FASCINATION/outputs/visualisation/metrics_{season}_pca_metrics.csv")

if __name__ == "__main__":
    main()

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
import gc

from FASCINATION.src.utils import get_min_max_idx, get_f1_score, cubic_interpolate_along_axis, compute_psnr, compute_msssim
from itertools import combinations
import pickle

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

    explained_variance_thresholds = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999] #  0.9, 0.95, 0.98, 0.99, 0.999
    season_metrics = {}

    pca_types = ["depth", "spatial", "time"] #"time" #
    pca_combinations = []
    for i in range(1, 4):
        pca_combinations.extend(combinations(pca_types, i))

    for season in tqdm(["all", "spring", "summer", "autumn", "winter"], desc="Processing seasons", unit="season"): #   "all", ,
        print(f"\nProcessing season: {season}")

        if season == "all":
            # Extract month and day for each time point
            natl_month_day = pd.DataFrame({
                "month": da_natl['time.month'].values,
                "day": da_natl['time.day'].values
            })
            enatl_month_day = pd.DataFrame({
                "month": da_enatl['time.month'].values,
                "day": da_enatl['time.day'].values
            })

            # Find common (month, day) pairs
            natl_md_set = set(tuple(x) for x in natl_month_day.values)
            enatl_md_set = set(tuple(x) for x in enatl_month_day.values)
            common_md = np.array(list(natl_md_set & enatl_md_set))

            # If there are fewer than 90, use all; else, sample 90
            n_sample = min(90, len(common_md))
            rng = np.random.default_rng(seed=42)
            selected_md = common_md[rng.choice(len(common_md), n_sample, replace=False)]

            # Get indices for natl and enatl matching the selected (month, day)
            natl_idx = natl_month_day.apply(lambda row: any((row["month"] == md[0]) and (row["day"] == md[1]) for md in selected_md), axis=1)
            enatl_idx = enatl_month_day.apply(lambda row: any((row["month"] == md[0]) and (row["day"] == md[1]) for md in selected_md), axis=1)

            da_natl_season = da_natl.isel(time=np.where(natl_idx)[0]).dropna(dim="time")
            da_enatl_season = da_enatl.isel(time=np.where(enatl_idx)[0]).dropna(dim="time")
        else:
            da_natl_season = da_natl.sel(time=da_natl['season'] == season).dropna(dim="time")
            da_enatl_season = da_enatl.sel(time=da_enatl['season'] == season).dropna(dim="time")

        min_time = min(da_natl_season.sizes["time"], da_enatl_season.sizes["time"])
        da_natl_season = da_natl_season.isel(time=slice(0, min_time))
        da_enatl_season = da_enatl_season.isel(time=slice(0, min_time))

        natl_arr = da_natl_season.values
        enatl_arr = da_enatl_season.values

        t, z, lat, lon = natl_arr.shape
        season_metrics[season] = {}

        for pca_combo in tqdm(pca_combinations, desc="Processing PCA combinations", unit="combo"):
            combo_name = "_".join(pca_combo)
            season_metrics[season][combo_name] = {}
            for threshold in tqdm(explained_variance_thresholds, desc=f"Processing thresholds ({combo_name})", unit="threshold"):
                arr_natl = natl_arr.copy()
                arr_enatl = enatl_arr.copy()
                shape = (t, z, lat, lon)
                pca_objs = {}
                n, m, k = z, lat * lon, t  # default values

                # Apply PCA(s) in order
                for pca_type in pca_combo:
                    if pca_type == "depth":
                        depth_pca = PCA()
                        depth_pca.fit(arr_natl.transpose(0, 2, 3, 1).reshape(-1, z))
                        cumulative_variance_depth = np.cumsum(depth_pca.explained_variance_ratio_)
                        n = np.argmax(cumulative_variance_depth >= threshold) + 1

                        depth_pca = PCA(n_components=n)
                        natl_depth_pca = depth_pca.fit_transform(arr_natl.transpose(0, 2, 3, 1).reshape(-1, z))
                        enatl_depth_pca = depth_pca.transform(arr_enatl.transpose(0, 2, 3, 1).reshape(-1, z))

                        arr_natl = natl_depth_pca.reshape(t, lat, lon, n).transpose(0, 3, 1, 2)
                        arr_enatl = enatl_depth_pca.reshape(t, lat, lon, n).transpose(0, 3, 1, 2)
                        pca_objs["depth"] = depth_pca
                    elif pca_type == "spatial":
                        arr_shape = arr_natl.shape
                        lat_spatial = arr_shape[2]
                        lon_spatial = arr_shape[3]
                        spatial_pca = PCA()
                        spatial_pca.fit(arr_natl.reshape(-1, lat_spatial * lon_spatial))
                        cumulative_variance_spatial = np.cumsum(spatial_pca.explained_variance_ratio_)
                        m = np.argmax(cumulative_variance_spatial >= threshold) + 1


                        spatial_pca = PCA(n_components=m)
                        natl_spatial_pca = spatial_pca.fit_transform(arr_natl.reshape(-1, lat_spatial * lon_spatial))
                        enatl_spatial_pca = spatial_pca.transform(arr_enatl.reshape(-1, lat_spatial * lon_spatial))

                        arr_natl = natl_spatial_pca.reshape(t, n, m)
                        arr_enatl = enatl_spatial_pca.reshape(t, n, m)
                        pca_objs["spatial"] = spatial_pca


                    elif pca_type == "time":
                        arr_shape = arr_natl.shape
                        if len(arr_shape) == 2:
                            pass
                        
                        elif len(arr_shape) == 3:
                            arr_natl_time = arr_natl.transpose(1, 2, 0).reshape(-1, t)  
                            arr_enatl_time = arr_enatl.transpose(1, 2, 0).reshape(-1,t) 

                        else:
                            arr_natl_time = arr_natl.transpose(1, 2, 3, 0).reshape(-1, t)
                            arr_enatl_time = arr_enatl.transpose(1, 2, 3, 0).reshape(-1,t)

                        time_pca = PCA()
                        time_pca.fit(arr_natl_time)
                        cumulative_variance_time = np.cumsum(time_pca.explained_variance_ratio_)
                        k = np.argmax(cumulative_variance_time >= threshold) + 1

                        time_pca = PCA(n_components=k)
                        natl_time_pca = time_pca.fit_transform(arr_natl_time)
                        enatl_time_pca = time_pca.transform(arr_enatl_time)

                        arr_natl = natl_time_pca
                        arr_enatl = enatl_time_pca
                        pca_objs["time"] = time_pca

                # Inverse transform for reconstruction (on eNATL)
                arr_pred = arr_enatl
                # Reverse order for inverse transform
                for pca_type in reversed(pca_combo):
                    if pca_type == "time":
                        arr_shape = arr_natl.shape
                        arr_pred = pca_objs["time"].inverse_transform(arr_pred)
                        # reshape back to (t, n, m) or (t, n) or (t, m) depending on previous steps
                        if "spatial" in pca_combo:
                            arr_pred = arr_pred.reshape(n, m, t).transpose(2, 0, 1)
                        else:
                            arr_pred = arr_pred.reshape(n, lat, lon, t).transpose(3, 0, 1, 2)

                    elif pca_type == "spatial":
                        arr_shape = arr_pred.shape
                        arr_pred = pca_objs["spatial"].inverse_transform(arr_pred)
                        arr_pred = arr_pred.reshape(t, n, lat, lon)

                    elif pca_type == "depth":
                        arr_shape = arr_pred.shape
                        # arr_pred is always (t, n, lat, lon) after depth PCA
                        arr_pred = pca_objs["depth"].inverse_transform(arr_pred.transpose(0,2,3,1).reshape(-1, n))
                        arr_pred = arr_pred.reshape(t, lat, lon, z).transpose(0, 3, 1, 2)

                # For metrics, try to reshape arr_pred to arr_truth shape
                arr_truth = enatl_arr

                assert arr_truth.shape == arr_pred.shape, f"arr_truth shape {arr_truth.shape} does not match arr_pred shape {arr_pred.shape}"

                original_size = arr_truth.nbytes
                reduced_size = arr_enatl.nbytes if arr_enatl is not None and hasattr(arr_enatl, 'nbytes') else 1
                if "time" in pca_combo:
                    reduced_size = arr_enatl.nbytes if hasattr(arr_enatl, 'nbytes') else 1
                    if isinstance(arr_enatl, np.ndarray):
                        reduced_size = arr_enatl.nbytes
                    elif isinstance(arr_enatl, (list, tuple)):
                        reduced_size = np.array(arr_enatl).nbytes
                    else:
                        reduced_size = 1
                cr = original_size / reduced_size if reduced_size > 0 else np.nan

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
                season_metrics[season][combo_name][threshold] = {
                    "CR": cr,
                    "RMSE": ae_ssp_rmse,
                    "PSNR": psnr,
                    "MS-SSIM": msssim,
                    "ECS": ae_ecs_rmse,
                    "MAE": mae,
                    "mean_error_n_min_max": mean_error_n_min_max,
                    "F1_score": f1_score,
                    "R2_score": r2,
                    "n_depth": n if "depth" in pca_combo else None,
                    "m_spatial": m if "spatial" in pca_combo else None,
                    "k_time": k if "time" in pca_combo else None
                }
                # Memory cleanup after each threshold
                del arr_natl, arr_enatl, arr_pred, arr_truth, min_max_idx_truth, min_max_idx_pred, F1_score
                gc.collect()
                # Save progress after each threshold
                with open("/Odyssey/private/o23gauvr/code/FASCINATION/pickle/season_metrics_all_pca.pkl", "wb") as f:
                    pickle.dump(season_metrics, f)
    
    # Save season_metrics as a pickle file
    with open("/Odyssey/private/o23gauvr/code/FASCINATION/pickle/season_metrics_all_pca.pkl", "wb") as f:
        pickle.dump(season_metrics, f)
    # Save all metrics to CSV
    # Combine all results into a single DataFrame with MultiIndex
    all_results = []
    for season in season_metrics:
        for combo_name in season_metrics[season]:
            for threshold, metrics in season_metrics[season][combo_name].items():
                row = {
                    "combo_name": combo_name,
                    "threshold": threshold,
                    "season": season,
                }
                row.update(metrics)
                all_results.append(row)
    df_all = pd.DataFrame(all_results)
    # Set MultiIndex: (combo_name, threshold), columns: (season, metric)
    df_all = df_all.set_index(["combo_name", "threshold", "season"])
    #df_all = df_all.unstack(level="season")
    df_all.to_csv("/Odyssey/private/o23gauvr/code/FASCINATION/outputs/visualisation/complete_metrics_all_pca.csv")

if __name__ == "__main__":
    main()

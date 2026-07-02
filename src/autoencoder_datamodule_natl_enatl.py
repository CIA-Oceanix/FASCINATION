import sys
from tabnanny import verbose
import pytorch_lightning as pl
import xarray as xr
import numpy as np
import random
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple
from typing import Union, Tuple, List, Dict
import torch
import pandas as pd
import pickle
from scipy.signal import butter, filtfilt

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])


def get_seasonal_time_indices(da_time, season: str) -> np.ndarray:
    """
    Get time indices for a specific season.

    Parameters
    ----------
    da_time : array-like
        Time coordinate from an xarray DataArray.
    season : str
        One of: 'all', 'spring', 'summer', 'autumn', 'winter'.

    Returns
    -------
    np.ndarray
        Integer indices of timesteps belonging to the requested season.
    """
    if hasattr(da_time, 'values'):
        time_vals = pd.to_datetime(da_time.values)
    else:
        time_vals = pd.to_datetime(da_time)

    months = time_vals.month

    if season == 'all':
        return np.arange(len(time_vals))
    elif season == 'spring':
        return np.where((months >= 3) & (months <= 5))[0]
    elif season == 'summer':
        return np.where((months >= 6) & (months <= 8))[0]
    elif season == 'autumn':
        return np.where((months >= 9) & (months <= 11))[0]
    elif season == 'winter':
        return np.where((months == 12) | (months == 1) | (months == 2))[0]
    else:
        raise ValueError(
            f"Unknown season: {season!r}. Must be one of: "
            "'all', 'spring', 'summer', 'autumn', 'winter'"
        )


# Map month → season index
def month_to_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

def _seed_worker(worker_id):
    """Seed numpy/random in each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AEDatamodule(pl.LightningDataModule):
    """
    Datamodule that accepts separate train_da and test_da (xarray.DataArray).
    Both DAs are processed independently for nan management, factor_64 reshape, rgb depth_layers,
    and normalization. PCA and CAE-based rgb encoding have been removed.
    Only train and test splits/datasets are provided (no validation).
    """

    def __init__(
        self,
        dl_kw,
        norm_stats,
        data_type = "ssp",
        test_norm: str = "on_train",
        manage_nan: str = "supress_with_max_depth",
        reshape=None,
        rgb={"use": False, "method": None},
        dtype_str='float32',
        days_split={"method": "ratio", "value": (0.5, 0.5)},  # split test_da into val/test with this ratio
        shuffle: bool = True,
        uniform_z: bool = False,
        filtering: bool = False,
        normalize_per_split: bool = False,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        train_da, test_da : xarray.DataArray
            Input arrays with dims ("time","z","lat","lon"). They can have different
            lat/lon coords; z coords must be compatible (same depth axis semantics).
        dl_kw : dict
            dataloader keyword args (batch_size, num_workers, ...)
        norm_stats : dict
            e.g. {"method":"min_max", "params": None} or {"method":"mean_std", "params": None}
        manage_nan : str
            options: "suppress", "before_normalization", "supress_with_max_depth", None
        reshape : list or None
            include "factor_64" to pad/interpolate lat/lon to multiples of 64
        rgb : dict
            {"use": bool, "method": "depth_layers" or None}
            NOTE: CAE method removed.
        dtype_str : str
            numpy dtype string, e.g. 'float32'
        normalize_per_split : bool
            If True, compute normalization parameters separately for each split (train, val, test).
            If False, use test_norm parameter to control normalization behavior. Default: False.
        """
        super().__init__()

        if data_type=="ssp":
            data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                        "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}
        elif data_type=="temp":
            data_path ={"enatl": "/Odyssey/public/enatl60/raw/eNATL60_BLB002_degraded_votemper_regrid_0_botm.nc",
                        "natl": "/Odyssey/public/natl60/raw/NATL60GULF-CJM165_degraded_votemper_regrid.nc"}

        sst_path = {"enatl": "/Odyssey/public/enatl60/sst/eNATL60-BLB002-SST-2009-2010-1_20.nc",
                    "natl": "/Odyssey/public/natl60/sst/NATL60-CJM165-SST-2009-2010-1_20.nc"}

        self.train_da = xr.open_dataarray(data_path['enatl'])
        self.val_da=None
        self.test_da = xr.open_dataarray(data_path['natl'])

        self.train_sst = xr.open_dataarray(sst_path['enatl'])
        self.test_sst = xr.open_dataarray(sst_path['natl'])

        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        self.test_norm = test_norm
        self.normalize_per_split = normalize_per_split
        self.test_norm_stats = {"method": norm_stats.get("method", None), "params": None}
        self.rgb = rgb
        self.manage_nan = manage_nan
        self.filtering = filtering
        self.filter_cutoff = 0.1  # cutoff frequency for low-pass Butterworth
        self.n_profiles = None
        self.train_time_ratio = 1.0
        # self.val_time_ratio = 0.5
        # self.test_time_ratio = 0.5
        self.reshape = [] if reshape is None else reshape
        self.dtype_str = dtype_str
        self.days_split = days_split
        self.time_ratio = 0.1
        self.seed = seed
        self.shuffle = shuffle

        self.uniform_z = uniform_z
        self.depth_array = None

        # internal placeholders filled in setup
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.drop_last_batch = False
        
        # Normalization stats storage (computed during setup)
        self.train_norm_stats = None
        self.val_norm_stats = None
        self.test_norm_stats = None

        self.verbose = True

        # if rgb is requested, enforce min_max norm method for consistency (same behavior as before)
        if self.rgb["use"]:
            self.norm_stats["method"] = "min_max"



    def _ensure_dtype(self, da: xr.DataArray):
        required_dtype = getattr(np, self.dtype_str)
        if da.dtype != required_dtype:
            return da.astype(required_dtype)
        return da

    def _manage_nan_single_da(self, da: xr.DataArray):
        """Apply manage_nan rules to a single DataArray (returns processed da)."""
        if self.manage_nan is None:
            return da

        if self.manage_nan == "suppress":
            # drop lat coords containing any NaN across remaining dims
            return da.dropna(dim="lat")
        elif self.manage_nan == "before_normalization":
            return da.fillna(0)
        elif self.manage_nan == "supress_with_max_depth":
            max_depth = 2000
            # Keep only z < max_depth, then drop latitudes that have any NaN across time,z,lon
            sub_da = da.sel(z=da.z.where(da.z < max_depth, drop=True))
            lat_nan = sub_da.isnull().any(dim=["time", "z", "lon"])
            valid_lats = lat_nan.where(lat_nan == False, drop=True).coords["lat"].values
            # If no valid lats, return original to avoid empty dataset (caller may handle)
            if len(valid_lats) == 0:
                return da
            return da.sel(lat=valid_lats, z=da.z.where(da.z < max_depth, drop=True))
        else:
            # unknown option: no-op
            return da

    # def _spatio_temporal_subsample(self, da: xr.DataArray, n_profiles):
    #     """Attempt to respect n_profiles by subsampling time/space similar to original logic, applied per-DA."""
    #     if n_profiles is None:
    #         return da

    #     time_size = len(da.time)
    #     lat_size = len(da.lat)
    #     lon_size = len(da.lon)

    #     # adapt space_ratio so we get a reasonable time_factor similar to original behavior
    #     while True:
    #         space_factor = max(1, int(round(1 / self.space_ratio)))
    #         time_factor = max(
    #             1,
    #             int(time_size * np.ceil(lat_size / space_factor) * np.ceil(lon_size / space_factor)) // max(1, self.n_profiles)
    #         )
    #         if time_factor <= time_size / 10:
    #             break
    #         self.space_ratio *= 0.5
    #         # if space_ratio becomes tiny, break to avoid infinite loop
    #         if self.space_ratio < 1e-6:
    #             break

    #     # apply subsampling
    #     lat_slice = slice(0, None, space_factor)
    #     lon_slice = slice(0, None, space_factor)
    #     time_slice = slice(0, None, max(1, int(time_factor)))
    #     return da.isel(time=time_slice, lat=lat_slice, lon=lon_slice)

    # def _select_days(self, da: xr.DataArray, days_ratio=None):
    #     if days_ratio is None:
    #         return da
    #     # Implement logic to select n_days from the DataArray
    #     # Placeholder: select first n_days
    #     step = int(1/days_ratio)
    #     return da.isel(time=slice(0, None, step))


    def _split_da_along_time(
        self,
        da: xr.DataArray,
        days_split: Dict,
    ) -> Union[xr.DataArray, List[xr.DataArray]]:
        """
        Subsample or split a DataArray along the time dimension.

        Parameters
        ----------
        da : xr.DataArray
            Input data with a 'time' dimension.
        days_ratio : dict
            A dict with keys ``method`` and ``value``:

            - ``{"method": "subsample", "value": float}``
                Subsample time with step = int(1 / value).  value must be in (0, 1].

            - ``{"method": "ratio", "value": tuple of floats}``
                Return len(value) contiguous blocks whose sizes are proportional to
                the ratios and separated by n_gap.

            - ``{"method": "season", "value": list of season specs}``
                Return one DataArray per element.  Each element is either a season
                string ('spring', 'summer', 'autumn', 'winter', 'all') or a tuple
                of season strings to merge.  Example: [("spring", "summer"), "winter"]

            - ``{"method": "alternate_days", "value": tuple of ints}``
                Alternate between len(value) splits, assigning value[i] consecutive
                days to split i, then skipping n_gap days, then moving to split i+1,
                etc., cycling until the end of the time axis.
                Example: value=(7, 60), n_gap=15 →
                  7 days → split 0, 15-day gap, 60 days → split 1, 15-day gap,
                  7 days → split 0, 15-day gap, …

            n_gap : int
                Number of timesteps skipped between blocks (used for ``ratio`` and
                ``alternate_days`` modes; for ``season`` mode it trims boundaries).

        Returns
        -------
        xr.DataArray or list[xr.DataArray]
        """

        if "time" not in da.dims:
            raise ValueError("DataArray must have a 'time' dimension")

        if not isinstance(days_split, dict) or "method" not in days_split or "value" not in days_split:
            raise TypeError(
                "days_split must be a dict with 'method' and 'value' keys. "
                "Supported methods: 'subsample', 'ratio', 'season', 'alternate_days'."
            )

        method = days_split["method"]
        value  = days_split["value"]
        n_gap = days_split.get("n_gap", 0)

        # ------------------------------------------------------------------
        # Case 1 — simple subsampling
        # ------------------------------------------------------------------
        if method == "subsample":
            if not isinstance(value, (int, float)) or not (0 < value <= 1):
                raise ValueError("subsample value must be a float in (0, 1]")
            step = int(1 / value)
            return da.isel(time=slice(0, None, step))

        # ------------------------------------------------------------------
        # Case 2 — contiguous block splits with gaps
        # ------------------------------------------------------------------
        if method == "ratio":
            ratios = list(value)
            n_sets = len(ratios)

            if any(r <= 0 for r in ratios):
                raise ValueError("All ratios must be positive")

            T = da.sizes["time"]
            T_eff = T - (n_sets - 1) * n_gap

            if T_eff <= 0:
                raise ValueError("n_gap too large for dataset length")

            # normalize ratios
            ratio_sum = sum(ratios)
            ratios = [r / ratio_sum for r in ratios]

            # block sizes
            block_sizes = [int(T_eff * r) for r in ratios]

            # ensure exact coverage (last block absorbs rounding)
            block_sizes[-1] += T_eff - sum(block_sizes)

            splits = []
            start = 0

            for size in block_sizes:
                end = start + size
                splits.append(da.isel(time=slice(start, end)))
                start = end + n_gap

            return splits

        # ------------------------------------------------------------------
        # Case 3 — season-based splits with temporal gap enforcement
        # ------------------------------------------------------------------
        if method == "season":
            days_ratio_list = value
            if not isinstance(days_ratio_list, list):
                raise TypeError("season value must be a list of season specs")
            # Step 1: build raw sorted index sets per split
            raw_indices: List[List[int]] = []
            for season_spec in days_ratio_list:
                if isinstance(season_spec, str):
                    seasons = [season_spec]
                elif isinstance(season_spec, tuple):
                    seasons = list(season_spec)
                else:
                    raise TypeError(
                        "Each element of the season list must be a str or a tuple of str, "
                        f"got {type(season_spec)}"
                    )
                combined: set = set()
                for s in seasons:
                    combined.update(get_seasonal_time_indices(da.time, s).tolist())
                raw_indices.append(sorted(combined))

            # Step 2: enforce n_gap at every temporal boundary between splits
            if n_gap > 0:
                time_vals = pd.to_datetime(da.time.values)
                gap_td = pd.Timedelta(days=n_gap)
                half1 = n_gap // 2       # days trimmed from the earlier-ending split
                half2 = n_gap - half1    # days trimmed from the later-starting split

                # Build a global sorted list of (time_index, split_id)
                tagged: List[tuple] = []
                for split_id, idx_list in enumerate(raw_indices):
                    for idx in idx_list:
                        tagged.append((idx, split_id))
                tagged.sort(key=lambda x: x[0])

                to_remove: List[set] = [set() for _ in range(len(raw_indices))]

                # Walk through consecutive pairs; act on every cross-split boundary
                # whose timestamps are closer than n_gap days
                for k in range(len(tagged) - 1):
                    idx_k,  split_k  = tagged[k]
                    idx_k1, split_k1 = tagged[k + 1]

                    if split_k == split_k1:
                        continue

                    if (time_vals[idx_k1] - time_vals[idx_k]) >= gap_td:
                        continue  # already far enough apart

                    # Remove last half1 timesteps of split_k before this boundary
                    count, pos = 0, k
                    while pos >= 0 and count < half1:
                        if tagged[pos][1] == split_k:
                            to_remove[split_k].add(tagged[pos][0])
                            count += 1
                        pos -= 1

                    # Remove first half2 timesteps of split_k1 after this boundary
                    count, pos = 0, k + 1
                    while pos < len(tagged) and count < half2:
                        if tagged[pos][1] == split_k1:
                            to_remove[split_k1].add(tagged[pos][0])
                            count += 1
                        pos += 1

                final_indices = [
                    [i for i in idx_list if i not in to_remove[sid]]
                    for sid, idx_list in enumerate(raw_indices)
                ]
            else:
                final_indices = raw_indices

            return [da.isel(time=idx_list) for idx_list in final_indices]

        # ------------------------------------------------------------------
        # Case 4 — alternating day blocks
        # ------------------------------------------------------------------
        if method == "alternate_days":
            block_sizes = list(value)
            if self.rgb['use']:
                day_block = len(self.depth_array)//3
                block_sizes = [day_block*i for i in block_sizes]
            n_splits = len(block_sizes)
            if n_splits < 2:
                raise ValueError("alternate_days value must contain at least 2 block sizes")
            if any(b <= 0 for b in block_sizes):
                raise ValueError("All block sizes in alternate_days must be positive integers")

            T = da.sizes["time"]
            split_indices: List[List[int]] = [[] for _ in range(n_splits)]

            pos = 0
            split_turn = 0
            while pos < T:
                block = block_sizes[split_turn]
                end = min(pos + block, T)
                split_indices[split_turn].extend(range(pos, end))
                pos = end + n_gap
                split_turn = (split_turn + 1) % n_splits

            return [da.isel(time=idx_list) for idx_list in split_indices]

        raise ValueError(
            f"Unknown method {method!r}. "
            "Must be one of: 'subsample', 'ratio', 'season', 'alternate_days'."
        )

    def _factor_64_pad_interp(self, da: xr.DataArray):
        """If 'factor_64' in reshape: interpolate lat/lon so sizes are multiples of 64.
           Interpolation is done per-DA independently (so resulting lat/lon may differ between train/test)."""
        if "factor_64" not in self.reshape:
            return da

        lat_size = len(da.lat)
        lon_size = len(da.lon)
        n_lat = int(np.ceil(lat_size / 64))
        closest_lat_size = n_lat * 64
        n_lon = int(np.ceil(lon_size / 64))
        closest_lon_size = n_lon * 64
        if closest_lat_size == lat_size and closest_lon_size == lon_size:
            return da

        new_lat = np.linspace(da.lat.min().item(), da.lat.max().item(), closest_lat_size)
        new_lon = np.linspace(da.lon.min().item(), da.lon.max().item(), closest_lon_size)
        # Use cubic interpolation as in original code
        return da.interp(lat=new_lat, lon=new_lon, method="cubic")

    def _rgb_depth_layers(self, da: xr.DataArray):
        """If rgb.use and rgb.method == 'depth_layers', pack depth dimension into RGB-like channels.
           Strategy: split z into groups of 3 channels (or fewer if depth not multiple of 3) and
           reshape time accordingly (similar to original).
           This function returns a new DataArray with z dimension = 3 and longer time dimension.
           NOTE: this operation changes the meaning of time → ensure this is desired.
        """
        if not (self.rgb.get("use", False) and self.rgb.get("method") == "depth_layers"):
            return da

        data = da.data  # (time, z, lat, lon)
        n_channels = 3
        z_dim = data.shape[1]
        n_rgb = z_dim // n_channels
        if n_rgb == 0:
            # not enough depth levels to form a single RGB triplet; return original
            return da

        # trim extra depth levels if not divisible by 3
        if n_rgb * n_channels != z_dim:
            data = data[:, :n_rgb * n_channels, :, :]

        # reshape: (time, n_rgb, n_channels, lat, lon) -> combine first two dims into new time
        data = data.reshape(data.shape[0], n_rgb, n_channels, data.shape[2], data.shape[3])
        data = data.reshape(-1, n_channels, data.shape[3], data.shape[4])

        # new coords: time is a RangeIndex, z becomes [1,2,3]
        new_time = pd.RangeIndex(data.shape[0], name="time")
        new_z = np.arange(1, n_channels + 1)
        new_da = xr.DataArray(
            data=data,
            dims=("time", "z", "lat", "lon"),
            coords={
                "time": new_time,
                "z": new_z,
                "lat": da.lat,
                "lon": da.lon
            },
            attrs=da.attrs.copy()
        )
        return new_da

    def _get_train_norm_stats(self, arr: np.array, norm_stats, verbose=False):
        """Compute norm stats from numpy array train_arr (numpy array)."""

        norm_stats['params'] = {}

        # method = self.norm_stats.get('method', None)
        # if method == "mean_std":
        norm_stats["params"]["mean"] = np.nanmean(arr)
        norm_stats["params"]["std"] = np.nanstd(arr)

        #elif method == "mean_std_along_depth":
            # mean/std along (time,lat,lon) per depth
            # Expect train_arr shape: (time, z, lat, lon)
        norm_stats["params"]["mean_along_depth"] = np.nanmean(arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        norm_stats["params"]["std_along_depth"] = np.nanstd(arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
                                                             
        #elif method == "min_max":
        norm_stats["params"]["x_min"] = np.nanmin(arr)
        norm_stats["params"]["x_max"] = np.nanmax(arr)
        # else:
        #     raise RuntimeError(f"Unknown normalization method: {method}")

        if verbose:
            print("Norm stats", norm_stats)


    def _apply_normalization_to_data(self, data: np.ndarray, norm_stats=None):
        """Apply normalization in-place to numpy array data. Expected shapes: (time,z,lat,lon) or similar."""
        method = norm_stats.get("method", None)
        params = norm_stats.get("params", None)
        if params is None:
            raise RuntimeError("Normalization params not computed yet (call get_train_norm_stats first)")

        if method == "min_max":
            x_min = params["x_min"]
            x_max = params["x_max"]
            return (data - x_min) / (x_max - x_min)
        elif method == "mean_std":
            mean = params["mean"]
            std = params["std"]
            return (data - mean) / std
        elif method == "mean_std_along_depth":
            mean = params["mean_along_depth"]
            std = params["std_along_depth"]
            return (data - mean) / std
        else:
            raise RuntimeError(f"Unknown normalization method: {method}")

    def _attach_sst(self, train_da: xr.DataArray, val_da: xr.DataArray, test_da: xr.DataArray):

        train_sst = self.train_sst.sel(time=train_da.time).astype(getattr(np, self.dtype_str))
        val_sst = self.test_sst.sel(time=val_da.time).astype(getattr(np, self.dtype_str))
        test_sst = self.test_sst.sel(time=test_da.time).astype(getattr(np, self.dtype_str))
        
        train_sst = train_sst.interp_like(train_da)
        val_sst = val_sst.interp_like(val_da)
        test_sst = test_sst.interp_like(test_da)

        # Compute mean and std from train SST
        sst_mean = train_sst.mean()
        sst_std = train_sst.std()

        # Normalize SST arrays
        train_sst = (train_sst - sst_mean) / sst_std
        val_sst = (val_sst - sst_mean) / sst_std
        test_sst = (test_sst - sst_mean) / sst_std

        train_da.attrs['sst'] = train_sst
        val_da.attrs['sst'] = val_sst
        test_da.attrs['sst'] = test_sst

        return train_da, val_da, test_da


    def _uniform_depth(self, da: xr.DataArray):
        """If uniform_z is True, interpolate along z to a uniform depth grid defined by self.depth_array."""
        if not self.uniform_z:
            return da

        z_uniform = np.linspace(float(da.z.min()), float(da.z.max()), len(da.z))
        return da.interp(z=z_uniform)


    def setup(self, stage=None):
        """
        stage is ignored in this simplified DM: we process train and test when setup is called.
        """
        required_dtype = getattr(np, self.dtype_str)

        # Work on copies to avoid modifying user-provided DAs
        train_da = self.train_da.copy().astype(required_dtype)
        test_da = self.test_da.copy().astype(required_dtype)

        # # Ensure dtypes for originals
        # if self.verbose:
        #     print("Ensuring dtypes for train/test DAs...")
        # self.train_da = self._ensure_dtype(train_da)
        # self.test_da = self._ensure_dtype(test_da)

        # # Save original_data attribute separately per DA
        # train_da.attrs["original_data"] = train_da.copy()
        # test_da.attrs["original_data"] = test_da.copy()

        # 1) NAN management per DA
        if self.verbose:
            print("Managing NaNs for train/test DAs...")
        train_da = self._manage_nan_single_da(train_da)
        test_da = self._manage_nan_single_da(test_da)

        # Uniform depth
        if self.uniform_z:
            train_da = self._uniform_depth(train_da)
            test_da = self._uniform_depth(test_da)


        self.depth_array = train_da.z.values.copy()

        # After nan management, ensure depth coords exist and are comparable (we only require same depth axis semantics)
        # We do NOT force lat/lon alignment between train/test.

        train_lat_lon = {'lat': train_da.lat.copy(), 'lon': train_da.lon.copy()}
        val_lat_lon = {'lat': test_da.lat.copy(), 'lon': test_da.lon.copy()}
        test_lat_lon = {'lat': test_da.lat.copy(), 'lon': test_da.lon.copy()}


        # 3) factor_64 interpolation separately per DA
        if self.verbose:
            print("Applying factor_64 padding/interpolation for train/test DAs...")
        train_da = self._factor_64_pad_interp(train_da)
        test_da = self._factor_64_pad_interp(test_da)

        # 4) rgb depth_layers option (applied per DA)

        if self.rgb.get("use", False):
            if self.verbose:
                print("Applying RGB depth_layers method for train/test DAs")
            if self.rgb.get("method") == "depth_layers":
                train_da = self._rgb_depth_layers(train_da)
                test_da = self._rgb_depth_layers(test_da)
            else:
                # CAE-based rgb method removed — raise error if user requests it
                raise RuntimeError("rgb method 'CAE' is removed. Only 'depth_layers' is supported if rgb.use is True.")


        # Filtering low band
        if self.filtering:
            if self.verbose:
                print("Applying low-pass Butterworth filter to train/test DAs...")
            # Design Butterworth filter
            #b, a = butter(N=4, Wn=self.filter_cutoff, btype='low', fs=1.0)
            b, a = butter(N=2, Wn=self.filter_cutoff, btype='low', analog=False)
            # Apply filter along time axis for each depth/lat/lon point
            train_da.data = filtfilt(b, a, train_da.data, axis=0)
            test_da.data = filtfilt(b, a, test_da.data, axis=0)


        # 5) compute normalization statistics before splitting (for backward compatibility)
        # This is needed because test_norm="on_test" computes stats from the full test_da (before splitting)
        if not self.normalize_per_split and self.test_norm == "on_test":
            # Compute test_norm_stats from full test_da before splitting
            if self.verbose:
                print("Computing normalization stats from full test DA (before splitting)...")
            self.test_norm_stats = self.norm_stats.copy()
            self.test_norm_stats["params"] = {}
            self._get_train_norm_stats(test_da.data, self.test_norm_stats, verbose=False)
            self.test_norm_stats["norm_from"] = "full_natl"

        # 6) spatio-temporal subsample based on n_profiles (per DA) - BEFORE normalization
        if self.verbose:
            print("Selecting days for train/test DAs...")
        train_da = self._split_da_along_time(train_da, days_split={"method": "subsample", "value": self.train_time_ratio})
        val_da, test_da = self._split_da_along_time(test_da, days_split=self.days_split)  #{"method": "season", "value": ["summer", ("autumn", "winter", "spring")]}

        # 7) compute and apply normalization after split
        if self.verbose:
            print(f"Computing and applying normalization (normalize_per_split={self.normalize_per_split})...")
        
        if self.normalize_per_split:
            # Compute normalization stats separately for each split
            self.train_norm_stats = self.norm_stats.copy()
            self.val_norm_stats = self.norm_stats.copy()
            self.test_norm_stats = self.norm_stats.copy()
            
            # Compute stats from each split
            train_arr = train_da.data
            if self.verbose:
                print("  Computing normalization stats from train split...")
            if self.train_norm_stats.get("params") is None or any(v is None for v in (self.train_norm_stats.get("params") or {}).values()):
                self._get_train_norm_stats(train_arr, self.train_norm_stats, verbose=False)
            self.train_norm_stats["norm_from"] = "train"
            
            val_arr = val_da.data
            if self.verbose:
                print("  Computing normalization stats from val split...")
            self.val_norm_stats["params"] = {}
            self._get_train_norm_stats(val_arr, self.val_norm_stats, verbose=False)
            self.val_norm_stats["norm_from"] = "val"
            
            test_arr = test_da.data
            if self.verbose:
                print("  Computing normalization stats from test split...")
            self.test_norm_stats["params"] = {}
            self._get_train_norm_stats(test_arr, self.test_norm_stats, verbose=False)
            self.test_norm_stats["norm_from"] = "test"
            
            # Apply normalization to each split with its own stats
            train_da[:] = self._apply_normalization_to_data(train_da.data, self.train_norm_stats)
            train_da.attrs['norm_stats'] = self.train_norm_stats
            
            val_da[:] = self._apply_normalization_to_data(val_da.data, self.val_norm_stats)
            val_da.attrs['norm_stats'] = self.val_norm_stats
            
            test_da[:] = self._apply_normalization_to_data(test_da.data, self.test_norm_stats)
            test_da.attrs['norm_stats'] = self.test_norm_stats
        else:
            # Use the original test_norm logic for backward compatibility
            if self.verbose:
                print("  Computing normalization stats from train split...")
            train_arr = train_da.data
            if self.norm_stats.get("params") is None or any(v is None for v in (self.norm_stats.get("params") or {}).values()):
                self._get_train_norm_stats(train_arr, self.norm_stats, verbose=False)
            self.norm_stats["norm_from"] = "train"
            
            if self.test_norm == "on_train":
                self.test_norm_stats = self.norm_stats.copy()
                self.test_norm_stats["norm_from"] = "train"
            # else: test_norm == "on_test" - stats already computed above before splitting
            
            # Store for model access
            self.train_norm_stats = self.norm_stats
            self.val_norm_stats = self.test_norm_stats
            
            # Apply normalization
            if self.verbose:
                print("  Applying normalization to train/val/test splits...")
            train_da[:] = self._apply_normalization_to_data(train_da.data, self.norm_stats)
            val_da[:] = self._apply_normalization_to_data(val_da.data, self.test_norm_stats)
            test_da[:] = self._apply_normalization_to_data(test_da.data, self.test_norm_stats)



        # 8) assign normalized data back to DataArrays
        if self.verbose:
            print("Reconstructing normalized train/test DAs...")
        train_da = train_da.astype(required_dtype)
        val_da = val_da.astype(required_dtype)
        test_da = test_da.astype(required_dtype)



        # attach season_idx attributes if desired (original code did this)
        if self.verbose:
            print("Attaching season_idx and depth attributes to train/test DAs...")
        train_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(train_da["time"]).month.values]
        val_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(val_da["time"]).month.values]
        test_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(test_da["time"]).month.values]

        train_da.attrs['original_space_coords'] = train_lat_lon
        val_da.attrs['original_space_coords'] = val_lat_lon
        test_da.attrs['original_space_coords'] = test_lat_lon

        # norm_stats are already attached in step 6 if normalize_per_split, otherwise attach here
        if not self.normalize_per_split:
            train_da.attrs['norm_stats'] = self.norm_stats
            val_da.attrs['norm_stats'] = self.norm_stats if self.test_norm == "on_train" else self.test_norm_stats
            test_da.attrs['norm_stats'] = self.test_norm_stats if self.test_norm == "on_test" else self.norm_stats

        if self.rgb.get("use", False):
            if self.verbose:
                print("Attaching SST data to train/test DAs...")
            train_da, val_da, test_da = self._attach_sst(train_da, val_da, test_da)

        # store final processed DAs and create datasets
        if self.verbose:
            print("Storing final processed train/test DAs...")


        self.train_shape = train_da.shape
        self.val_shape = val_da.shape
        self.test_shape = test_da.shape

        

        # Create torch datasets (keeps ordering time,z,lat,lon and returns .data per sample)
        if self.verbose:
            print("Creating torch datasets for train/val/test DAs...")
        self.train_ds = AE_BaseDataset_3D(train_da)
        self.val_ds = AE_BaseDataset_3D(val_da)
        self.test_ds = AE_BaseDataset_3D(test_da)

        


        del self.train_da
        del self.test_da
        del self.val_da
        del self.train_sst
        del self.test_sst



    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return torch.utils.data.DataLoader(
            self.train_ds,
            shuffle=self.shuffle,
            worker_init_fn=_seed_worker,
            generator=g,
            **self.dl_kw
        )
    
    def val_dataloader(self):

        return torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            **self.dl_kw
        )

    def test_dataloader(self):

        return torch.utils.data.DataLoader(
            self.test_ds,
            shuffle=False,
            **self.dl_kw
        )


class AE_BaseDataset_3D(torch.utils.data.Dataset):
    def __init__(self, ipt: xr.DataArray):
        super().__init__()
        # Ensure ordering time,z,lat,lon
        self.input = ipt.transpose('time', 'z', 'lat', 'lon')

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, index):
        # returns numpy array (time slice's data) — the training loop can convert to torch.tensor as needed
        return self.input[index].data



if __name__ == "__main__":
        
        save_dm = True
        batch_size = 4
        rgb = {"use":False, "method":"CAE"} #PCA
        chn = "3" if rgb["use"] else "157"
        dm_path = f"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm_{chn}_196_256_norm_on_test.pkl"

        data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                    "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}


        datamodule = AEDatamodule(
            dl_kw={"batch_size": batch_size, "num_workers": 8},
            norm_stats={"method": "min_max"}, #, "params": {"mean": None, "std": None}  #"method":"min_max"
            test_norm="on_test", #on_test
            manage_nan="supress_with_max_depth",
            reshape=["factor_64"], #["factor_64"], #"RGB"
            rgb=rgb,
            uniform_z=True,
            dtype_str="float32",
            filtering=True,
            shuffle=True,
            normalize_per_split=True
            )

        datamodule.setup()

        # save_data = {
        #     'train_da': datamodule.train_da,
        #     'test_da': datamodule.test_da,
        #     'norm_stats': datamodule.norm_stats,
        #     'config': {
        #         'batch_size': batch_size,
        #         'rgb': rgb,
        #         'manage_nan': 'supress_with_max_depth',
        #         'reshape': ['factor_64']
        #     }
        # }

        if save_dm:
            print("Saving datamodule to:", dm_path)
            with open(dm_path, 'wb') as f:
                pickle.dump(datamodule, f)


import sys
from tabnanny import verbose
import pytorch_lightning as pl
import xarray as xr
import numpy as np
import random
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple
from typing import Union, Tuple, List
import torch
import pandas as pd
import pickle




TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

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
        data_name: str,
        dl_kw,
        norm_stats,
        manage_nan: str = "supress_with_max_depth",
        reshape={"factor_64": True, "spatial_crop": 5},  # set to {} or None to disable
        rgb={"use": False, "method": None},
        dtype_str='float32',
        space_ratio_init: float = 0.2,
        shuffle: bool = True,
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
            If False, compute from train split and apply to all splits. Default: False.
        """
        super().__init__()

        data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
                    "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc",
                    "natl_sst": "/Odyssey/public/natl60/raw/NATL60GULF-CJM165_degraded_vosaline_regrid.nc"}
        
        sst_path = {"enatl": "/Odyssey/public/enatl60/sst/eNATL60-BLB002-SST-2009-2010-1_20.nc",
                    "natl": "/Odyssey/public/natl60/sst/NATL60-CJM165-SST-2009-2010-1_20.nc",
                    "natl_sst": "/Odyssey/public/natl60/raw/NATL60GULF-CJM165_degraded_vosaline_regrid.nc"}
        
        self.data_name = data_name

        self.train_da = xr.open_dataarray(data_path[data_name])
        self.val_da=None
        self.test_da = None

        self.train_sst = xr.open_dataarray(sst_path[data_name])
        self.test_sst = None

        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        self.rgb = rgb
        self.manage_nan = manage_nan
        self.normalize_per_split = normalize_per_split
        self.n_profiles = None
        self.train_time_ratio = 0.7
        self.val_time_ratio = 0.1
        self.test_time_ratio = 0.3
        self.reshape = {} if reshape is None else reshape
        self.dtype_str = dtype_str
        self.space_ratio = space_ratio_init
        self.time_ratio = 0.1
        self.seed = seed
        self.shuffle = shuffle

        self.depth_array = None

        # internal placeholders filled in setup
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.drop_last_batch = False

        self.setup_called = False

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
        days_ratio: Union[float, Tuple[float, ...]],
        n_gap: int = 7,
    ) -> Union[xr.DataArray, List[xr.DataArray]]:
        """
        Subsample or split a DataArray along the time dimension.

        Parameters
        ----------
        da : xr.DataArray
            Input data with a 'time' dimension.
        days_ratio : float or tuple of floats
            - float: subsample time with step = int(1 / days_ratio)
            - tuple: return len(days_ratio) contiguous blocks whose sizes
                    are proportional to the ratios and separated by n_gap
        n_gap : int
            Number of timesteps separating consecutive blocks

        Returns
        -------
        xr.DataArray or list[xr.DataArray]
        """

        if "time" not in da.dims:
            raise ValueError("DataArray must have a 'time' dimension")

        # ------------------------------------------------------------------
        # Case 1 — simple subsampling
        # ------------------------------------------------------------------
        if isinstance(days_ratio, float):
            if not (0 < days_ratio <= 1):
                raise ValueError("days_ratio must be in [0, 1]")
            step = int(1 / days_ratio)
            return da.isel(time=slice(0, None, step))

        # ------------------------------------------------------------------
        # Case 2 — contiguous block splits with gaps
        # ------------------------------------------------------------------
        if isinstance(days_ratio, tuple):
            ratios = list(days_ratio)
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

        raise TypeError("days_ratio must be a float or a tuple of floats")

    def _factor_64_pad_interp(self, da: xr.DataArray):
        """If 'factor_64' in reshape: interpolate lat/lon so sizes are multiples of 64.
           Interpolation is done per-DA independently (so resulting lat/lon may differ between train/test)."""


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
        val_sst = self.train_sst.sel(time=val_da.time).astype(getattr(np, self.dtype_str))
        test_sst = self.train_sst.sel(time=test_da.time).astype(getattr(np, self.dtype_str))
        
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



    def setup(self, stage=None):
        """
        stage is ignored in this simplified DM: we process train and test when setup is called.
        """

        if self.setup_called:  
            return
        
        required_dtype = getattr(np, self.dtype_str)

        # Work on copies to avoid modifying user-provided DAs
        train_da = self.train_da.copy().astype(required_dtype)
        #test_da = self.test_da.copy().astype(required_dtype)

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
        #test_da = self._manage_nan_single_da(test_da)

        train_lat_lon = {'lat': train_da.lat.copy(), 'lon': train_da.lon.copy()}
        val_lat_lon = {'lat': train_da.lat.copy(), 'lon': train_da.lon.copy()}
        test_lat_lon = {'lat': train_da.lat.copy(), 'lon': train_da.lon.copy()}

        self.depth_array = train_da.z.values.copy()


        # After nan management, ensure depth coords exist and are comparable (we only require same depth axis semantics)
        # We do NOT force lat/lon alignorm_statsent between train/test.
        
        


        # 3) factor_64 interpolation separately per DA
        if self.verbose:
            print("Applying factor_64 padding/interpolation for train/test DAs...")




        if self.reshape.get("factor_64", False):
            train_da = self._factor_64_pad_interp(train_da)
        #test_da = self._factor_64_pad_interp(test_da)

        spatial_crop_idx = self.reshape.get("spatial_crop", 0)
        if spatial_crop_idx > 0:
            train_da = train_da.isel(lat=slice(spatial_crop_idx, -spatial_crop_idx), lon=slice(spatial_crop_idx, -spatial_crop_idx))

        # 4) rgb depth_layers option (applied per DA)

        if self.rgb.get("use", False):
            if self.verbose:
                print("Applying RGB depth_layers method for train/test DAs")
            if self.rgb.get("method") == "depth_layers":
                train_da = self._rgb_depth_layers(train_da)
                #test_da = self._rgb_depth_layers(test_da)
            else:
                # CAE-based rgb method removed — raise error if user requests it
                raise RuntimeError("rgb method 'CAE' is removed. Only 'depth_layers' is supported if rgb.use is True.")

        # 5) spatio-temporal subsample based on n_profiles (per DA)
        if self.verbose:
            print("Selecting days for train/test DAs...")
            train_da, val_da, test_da = self._split_da_along_time(train_da, days_ratio=(self.train_time_ratio,self.val_time_ratio,self.test_time_ratio), n_gap=7)

        # 6) compute and apply normalization after split
        if self.verbose:
            print(f"Computing and applying normalization (normalize_per_split={self.normalize_per_split})...")
        
        if self.normalize_per_split:
            # Compute normalization stats separately for each split
            train_norm_stats = self.norm_stats.copy()
            val_norm_stats = self.norm_stats.copy()
            test_norm_stats = self.norm_stats.copy()
            
            # Compute stats from each split
            train_arr = train_da.data
            if self.verbose:
                print("  Computing normalization stats from train split...")
            if train_norm_stats.get("params") is None or any(v is None for v in (train_norm_stats.get("params") or {}).values()):
                self._get_train_norm_stats(train_arr, train_norm_stats, verbose=False)
            train_norm_stats["norm_from"] = "train"
            
            val_arr = val_da.data
            if self.verbose:
                print("  Computing normalization stats from val split...")
            val_norm_stats["params"] = {}
            self._get_train_norm_stats(val_arr, val_norm_stats, verbose=False)
            val_norm_stats["norm_from"] = "val"
            
            test_arr = test_da.data
            if self.verbose:
                print("  Computing normalization stats from test split...")
            test_norm_stats["params"] = {}
            self._get_train_norm_stats(test_arr, test_norm_stats, verbose=False)
            test_norm_stats["norm_from"] = "test"
            
            # Apply normalization to each split with its own stats
            train_da[:] = self._apply_normalization_to_data(train_da.data, train_norm_stats)
            train_da.attrs['norm_stats'] = train_norm_stats
            
            val_da[:] = self._apply_normalization_to_data(val_da.data, val_norm_stats)
            val_da.attrs['norm_stats'] = val_norm_stats
            
            test_da[:] = self._apply_normalization_to_data(test_da.data, test_norm_stats)
            test_da.attrs['norm_stats'] = test_norm_stats
        else:
            # Compute normalization stats from train split and apply to all splits
            train_arr = train_da.data
            if self.verbose:
                print("  Computing normalization stats from train split (applied to all)...")
            if self.norm_stats.get("params") is None or any(v is None for v in (self.norm_stats.get("params") or {}).values()):
                self._get_train_norm_stats(train_arr, self.norm_stats, verbose=False)
            self.norm_stats["norm_from"] = "train"
            
            # Apply the same normalization to all splits
            if self.verbose:
                print("  Applying train normalization stats to all splits...")
            train_da[:] = self._apply_normalization_to_data(train_da.data, self.norm_stats)
            val_da[:] = self._apply_normalization_to_data(val_da.data, self.norm_stats)
            test_da[:] = self._apply_normalization_to_data(test_da.data, self.norm_stats)




        # 7) assign normalized data back to DataArrays
        if self.verbose:
            print("Reconstructing normalized train/val/test DAs...")
        train_da = train_da.astype(required_dtype)
        val_da = val_da.astype(required_dtype)
        test_da = test_da.astype(required_dtype)

        # attach season_idx attributes if desired (original code did this)
        if self.verbose:
            print("Attaching season_idx and depth attributes to train/val/test DAs...")
        train_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(train_da["time"]).month.values]
        val_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(val_da["time"]).month.values]
        test_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(test_da["time"]).month.values]

        train_da.attrs['original_space_coords'] = train_lat_lon
        val_da.attrs['original_space_coords'] = val_lat_lon
        test_da.attrs['original_space_coords'] = test_lat_lon

        # norm_stats are already attached in step 6 if normalize_per_split, otherwise attach here
        if not self.normalize_per_split:
            train_da.attrs['norm_stats'] = self.norm_stats
            val_da.attrs['norm_stats'] = self.norm_stats
            test_da.attrs['norm_stats'] = self.norm_stats


        if self.verbose and not self.rgb.get("use", False) and not "sst" in self.data_name:
            print("Attaching SST data to train/test DAs...")
            train_da, val_da, test_da = self._attach_sst(train_da, val_da, test_da)

        # store final processed DAs and create datasets
        if self.verbose:
            print("Storing final processed train/test DAs...")
        train_da = train_da
        val_da = val_da
        test_da = test_da

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

        self.setup_called = True


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
        
        save_dm = False
        data_name = "enatl"  #"natl"  #
        batch_size = 4
        rgb = {"use":False, "method":"CAE"} #PCA
        chn = "3" if rgb["use"] else "157"
        dm_path = f"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/{data_name}_dm_{chn}_196_256_good_split.pkl"


        datamodule = AEDatamodule(
            data_name=data_name,
            dl_kw={"batch_size": batch_size, "num_workers": 8},
            norm_stats={"method": "min_max"}, #, "params": {"mean": None, "std": None}  #"method":"min_max"on_test
            manage_nan="supress_with_max_depth",
            reshape={"factor_64": True, "spatial_crop": 0}, #["factor_64"], #"RGB"
            rgb=rgb,
            dtype_str="float32",
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


        # train_dataloader = datamodule.train_dataloader()
        # test_dataloader = datamodule.test_dataloader()



# def load_model(ckpt_path,input_shape=None,dl_kw=None, device='cpu', dtype_str='float32'):
#     cfg = get_cfg_from_ckpt_path(ckpt_path)
#     lit_mod = hydra.utils.call(cfg.model)
#     batch_shape = (dl_kw['batch_size'],*input_shape[1:]) 
#     lit_mod.model_hparams["input_shape"] = batch_shape
#     lit_mod.model_AE = AE_CNN(**lit_mod.model_hparams)
#     lit_mod.set_last_activation_function()
#     lit_mod.model_AE.to(device=device, dtype=dtype_str)

#     checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
#     checkpoint["state_dict"] = checkpoint["state_dict"]()  ##TODO supress this line after fixing saving issue
#     lit_mod.load_state_dict(checkpoint["state_dict"],strict=False)
#     lit_mod.norm_stats = checkpoint["norm_stats"]
#     lit_mod = lit_mod.eval()

#     for param in lit_mod.parameters():
#         param.requires_grad = False

#     return lit_mod
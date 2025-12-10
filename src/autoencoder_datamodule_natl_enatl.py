import pytorch_lightning as pl
import xarray as xr
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple
import torch
import pandas as pd

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


class AutoEncoderDatamodule_3D(pl.LightningDataModule):
    """
    Datamodule that accepts separate train_da and test_da (xarray.DataArray).
    Both DAs are processed independently for nan management, factor_64 reshape, rgb depth_layers,
    and normalization. PCA and CAE-based rgb encoding have been removed.
    Only train and test splits/datasets are provided (no validation).
    """

    def __init__(
        self,
        train_da: xr.DataArray,
        test_da: xr.DataArray,
        dl_kw,
        norm_stats,
        manage_nan: str = "supress_with_max_depth",
        n_profiles: int = None,
        reshape=None,
        rgb={"use": False, "method": None},
        dtype_str='float32',
        space_ratio_init: float = 0.2,
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
        """
        super().__init__()
        self.train_da_orig = train_da
        self.test_da_orig = test_da
        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        self.rgb = rgb
        self.manage_nan = manage_nan
        self.n_profiles = n_profiles
        self.reshape = [] if reshape is None else reshape
        self.dtype_str = dtype_str
        self.space_ratio = space_ratio_init
        self.time_ratio = 0.1
        self.seed = seed

        # internal placeholders filled in setup
        self.train_da = None
        self.test_da = None
        self.train_ds = None
        self.test_ds = None
        self.drop_last_batch = False
        self.is_data_normed = False

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

    def _spatio_temporal_subsample(self, da: xr.DataArray):
        """Attempt to respect n_profiles by subsampling time/space similar to original logic, applied per-DA."""
        if self.n_profiles is None:
            return da

        time_size = len(da.time)
        lat_size = len(da.lat)
        lon_size = len(da.lon)

        # adapt space_ratio so we get a reasonable time_factor similar to original behavior
        while True:
            space_factor = max(1, int(round(1 / self.space_ratio)))
            time_factor = max(
                1,
                int(time_size * np.ceil(lat_size / space_factor) * np.ceil(lon_size / space_factor)) // max(1, self.n_profiles)
            )
            if time_factor <= time_size / 10:
                break
            self.space_ratio *= 0.5
            # if space_ratio becomes tiny, break to avoid infinite loop
            if self.space_ratio < 1e-6:
                break

        # apply subsampling
        lat_slice = slice(0, None, space_factor)
        lon_slice = slice(0, None, space_factor)
        time_slice = slice(0, None, max(1, int(time_factor)))
        return da.isel(time=time_slice, lat=lat_slice, lon=lon_slice)

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

    def _get_train_norm_stats(self, train_arr: np.array, verbose=False):
        """Compute norm stats from numpy array train_arr (numpy array)."""
        self.norm_stats["params"] = {}
        method = self.norm_stats.get('method', None)
        if method == "mean_std":
            self.norm_stats["params"]["mean"] = np.nanmean(train_arr)
            self.norm_stats["params"]["std"] = np.nanstd(train_arr)
        elif method == "mean_std_along_depth":
            # mean/std along (time,lat,lon) per depth
            # Expect train_arr shape: (time, z, lat, lon)
            self.norm_stats["params"]["mean"] = np.nanmean(train_arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            self.norm_stats["params"]["std"] = np.nanstd(train_arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        elif method == "min_max":
            self.norm_stats["params"]["x_min"] = np.nanmin(train_arr)
            self.norm_stats["params"]["x_max"] = np.nanmax(train_arr)
        else:
            raise RuntimeError(f"Unknown normalization method: {method}")

        if verbose:
            print("Norm stats", self.norm_stats)
        return self.norm_stats

    def _apply_normalization_to_data(self, data: np.ndarray):
        """Apply normalization in-place to numpy array data. Expected shapes: (time,z,lat,lon) or similar."""
        method = self.norm_stats.get("method", None)
        params = self.norm_stats.get("params", None)
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
            mean = params["mean"]
            std = params["std"]
            return (data - mean) / std
        else:
            raise RuntimeError(f"Unknown normalization method: {method}")

    def setup(self, stage=None):
        """
        stage is ignored in this simplified DM: we process train and test when setup is called.
        """
        # Ensure dtypes for originals
        self.train_da_orig = self._ensure_dtype(self.train_da_orig)
        self.test_da_orig = self._ensure_dtype(self.test_da_orig)

        # Work on copies to avoid modifying user-provided DAs
        train_da = self.train_da_orig.copy()
        test_da = self.test_da_orig.copy()

        # Save original_data attribute separately per DA
        train_da.attrs["original_data"] = train_da.copy()
        test_da.attrs["original_data"] = test_da.copy()

        # 1) NAN management per DA
        train_da = self._manage_nan_single_da(train_da)
        test_da = self._manage_nan_single_da(test_da)

        # After nan management, ensure depth coords exist and are comparable (we only require same depth axis semantics)
        # We do NOT force lat/lon alignment between train/test.

        # 2) spatio-temporal subsample based on n_profiles (per DA)
        train_da = self._spatio_temporal_subsample(train_da)
        test_da = self._spatio_temporal_subsample(test_da)

        # 3) factor_64 interpolation separately per DA
        train_da = self._factor_64_pad_interp(train_da)
        test_da = self._factor_64_pad_interp(test_da)

        # 4) rgb depth_layers option (applied per DA)
        if self.rgb.get("use", False):
            if self.rgb.get("method") == "depth_layers":
                train_da = self._rgb_depth_layers(train_da)
                test_da = self._rgb_depth_layers(test_da)
            else:
                # CAE-based rgb method removed — raise error if user requests it
                raise RuntimeError("rgb method 'CAE' is removed. Only 'depth_layers' is supported if rgb.use is True.")

        # 5) compute normalization statistics from train only
        # prepare train numpy array for stats: ensure shape (time, z, lat, lon)
        train_arr = train_da.data  # numpy ndarray
        # if there are NaNs remaining, keep nan-aware stats
        if self.norm_stats.get("params") is None or any(v is None for v in (self.norm_stats.get("params") or {}).values()):
            self._get_train_norm_stats(train_arr)

        # 6) apply normalization to both DAs (use same params)
        train_data_normed = self._apply_normalization_to_data(train_da.data)
        test_data_normed = self._apply_normalization_to_data(test_da.data)

        # 7) assign normalized data back to DataArrays
        train_da = xr.DataArray(
            data=train_data_normed,
            dims=train_da.dims,
            coords=train_da.coords,
            attrs=train_da.attrs
        )
        test_da = xr.DataArray(
            data=test_data_normed,
            dims=test_da.dims,
            coords=test_da.coords,
            attrs=test_da.attrs
        )

        # 8) final housekeeping: dtype
        required_dtype = getattr(np, self.dtype_str)
        if train_da.dtype != required_dtype:
            train_da = train_da.astype(required_dtype)
        if test_da.dtype != required_dtype:
            test_da = test_da.astype(required_dtype)

        # attach season_idx attributes if desired (original code did this)
        train_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(train_da["time"]).month.values]
        test_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(test_da["time"]).month.values]

        # store final processed DAs and create datasets
        self.train_da = train_da
        self.test_da = test_da

        self.train_shape = self.train_da.shape
        self.test_shape = self.test_da.shape

        # Create torch datasets (keeps ordering time,z,lat,lon and returns .data per sample)
        self.train_ds = AE_BaseDataset_3D(self.train_da)
        self.test_ds = AE_BaseDataset_3D(self.test_da)

        # set drop_last behavior (unchanged)
        self.is_data_normed = True

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)

    # no validation loader (user requested to remove validation dataset)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)


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

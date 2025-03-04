import pytorch_lightning as pl
import xarray as xr
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple
from sklearn.decomposition import PCA
import torch
import os

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class AutoEncoderDatamodule_3D(pl.LightningDataModule):
    
    def __init__(self, input_da, dl_kw, norm_stats, pooled_dim:str, depth_pre_treatment: dict, manage_nan: str = "suppress", n_profiles: int = None, dtype_str = 'float32'):
        super().__init__()
        self.input = input_da
        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        self.manage_nan = manage_nan
        self.n_profiles = n_profiles
        self.depth_pre_treatment = depth_pre_treatment
        if pooled_dim != "spatial" or self.depth_pre_treatment["method"] != "pca":
            for key in self.depth_pre_treatment:
                self.depth_pre_treatment[key] = None
        self.dtype_str = dtype_str
        self.coords = input_da.coords
        self.depth_array = self.coords["z"].data
        self.input_shape = input_da.data.shape
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.drop_last_batch = False
        self.is_data_normed = False
        self.generator = torch.Generator().manual_seed(42)

    def setup(self, stage):
        required_dtype = getattr(np, self.dtype_str)
        if self.input.dtype != required_dtype:
            self.input = self.input.astype(required_dtype)

        if not self.is_data_normed:
            if self.manage_nan == "suppress":
                self.input = self.input.dropna(dim="lat")
            elif self.manage_nan == "before_normalization":
                self.input = self.input.fillna(0)
            elif self.manage_nan == "supress_with_max_depth":
                max_depth = 2000
                # Drop all lat coordinates presenting a nan for depths (z) inferior to 2000
                # Select only data for depths < 2000
                sub_da = self.input.sel(z=self.input.z.where(self.input.z < max_depth, drop=True))
                # For each lat, check if there is any nan across time, z, and lon
                lat_nan = sub_da.isnull().any(dim=["time", "z", "lon"])
                # Get valid latitudes (i.e. where there is no nan)
                valid_lats = lat_nan.where(lat_nan == False, drop=True).coords["lat"].values
                # Select only the valid latitudes and drop all z coordinates superior to 2000.
                self.input = self.input.sel(lat=valid_lats, z=self.input.z.where(self.input.z < max_depth, drop=True))
                
                self.coords = self.input.coords
                self.depth_array = self.coords["z"].data

            if self.n_profiles is not None:
                n_times = max(self.n_profiles // (len(self.input.lat) * len(self.input.lon)), 10)
            else:
                n_times = len(self.input.time)

            time_indices = torch.randint(0, len(self.input.time), (n_times,), generator=self.generator)
            train_size = int(0.7 * n_times)
            val_size = int(0.2 * n_times)

            self.train_time_idx = time_indices[:train_size]
            self.val_time_idx = time_indices[train_size:train_size + val_size]
            self.test_time_idx = time_indices[train_size + val_size:]

            train_data_array = self.input.isel(time=self.train_time_idx).data
            self.train_shape = self.input.isel(time=self.train_time_idx).shape
            self.test_shape = self.input.isel(time=self.test_time_idx).shape

            if self.depth_pre_treatment["norm_on"] == "components":
                input_shape = self.input.shape
                n_components = self.depth_pre_treatment["params"]
                pca = PCA(n_components=n_components, svd_solver='auto')
                train_data_array = pca.fit_transform(train_data_array.transpose(0, 2, 3, 1).reshape(-1, input_shape[1]))
                self.depth_pre_treatment["fitted_pca"] = pca
                self.get_train_norm_stats(train_data_array)
                data = pca.transform(self.input.data.transpose(0, 2, 3, 1).reshape(-1, input_shape[1])).reshape(input_shape[0], input_shape[2], input_shape[3], n_components).transpose(0, 3, 1, 2)
            else:
                data = self.input.data
                if any(param is None for param in self.norm_stats['params'].values()):
                    self.get_train_norm_stats(train_data_array)

            if self.norm_stats["method"] == "min_max":
                x_min = self.norm_stats["params"]["x_min"]
                x_max = self.norm_stats["params"]["x_max"]
                data = (data - x_min) / (x_max - x_min)
            elif self.norm_stats["method"] == "mean_std":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                data = (data - mean) / std
            elif self.norm_stats["method"] == "mean_std_along_depth":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                data = (data - mean) / std

            if self.depth_pre_treatment["method"] == "pca":
                if self.depth_pre_treatment["norm_on"] == "profiles":
                    input_shape = self.input.shape
                    n_components = self.depth_pre_treatment["params"]
                    pca = PCA(n_components=n_components, svd_solver='auto')
                    train_data_array = data[self.train_time_idx, :, :, :]
                    pca.fit(train_data_array.transpose(0, 2, 3, 1).reshape(-1, input_shape[1]))
                    self.depth_pre_treatment["fitted_pca"] = pca
                elif self.depth_pre_treatment["norm_on"] == "components":
                    data = pca.inverse_transform(data.transpose(0, 2, 3, 1).reshape(-1, n_components)).reshape(input_shape[0], input_shape[2], input_shape[3], input_shape[1]).transpose(0, 3, 1, 2)

            self.input.data = data

            self.min_val = self.input.data.min()
            self.max_val = self.input.data.max()

            # if self.depth_pre_treatment["method"] == "pca" and self.depth_pre_treatment["train_on"] == "components":
            #     coords = self.input.coords
            #     data = pca.transform(data.transpose(0, 2, 3, 1).reshape(-1, input_shape[1])).reshape(input_shape[0], input_shape[2], input_shape[3], n_components).transpose(0, 3, 1, 2)
            #     self.input = xr.DataArray(data=data, coords={"time": coords["time"], "z": np.arange(1, n_components + 1), "lat": coords["lat"], "lon": coords["lon"]})
            #     self.coords = self.input.coords

            if self.depth_pre_treatment["method"] == "pca":
                self.drop_last_batch = True

            self.is_data_normed = True

            if self.manage_nan == "after normalization":
                raise RuntimeError("a debugger, gérer pca + min")
                self.input = self.input.fillna(-6)

        if stage == 'fit':
            self.train_da = self.input.isel(time=self.train_time_idx)
            val_data_da = self.input.isel(time=self.val_time_idx)
            self.train_ds = AE_BaseDataset_3D(self.train_da)
            self.val_ds = AE_BaseDataset_3D(val_data_da)

        if stage == 'test':
            self.test_da = self.input.isel(time=self.test_time_idx)
            self.test_ds = AE_BaseDataset_3D(self.test_da)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)

    def get_train_norm_stats(self, train_arr: np.array, verbose=False):
        if self.norm_stats.method == "mean_std":
            self.norm_stats["params"]["mean"] = np.nanmean(train_arr)
            self.norm_stats["params"]["std"] = np.nanstd(train_arr)
        elif self.norm_stats.method == "mean_std_along_depth":
            if self.depth_pre_treatment["norm_on"] == "components":
                self.norm_stats["params"]["mean"] = np.nanmean(train_arr, axis=0).reshape(1, -1, 1, 1)
                self.norm_stats["params"]["std"] = np.nanstd(train_arr, axis=0).reshape(1, -1, 1, 1)
            else:
                self.norm_stats["params"]["mean"] = np.nanmean(train_arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
                self.norm_stats["params"]["std"] = np.nanstd(train_arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        elif self.norm_stats.method == "min_max":
            self.norm_stats["params"]["x_min"] = np.nanmin(train_arr)
            self.norm_stats["params"]["x_max"] = np.nanmax(train_arr)

        if verbose:
            print("Norm stats", self.norm_stats)

        return self.norm_stats
    

class AE_BaseDataset_3D(torch.utils.data.Dataset):
    def __init__(self, ipt):
        super().__init__()
        self.input = ipt.transpose('time', 'z', 'lat', 'lon')

    def __len__(self):
        return len(self.input.time)
    
    def __getitem__(self, index):
        return self.input[index].data





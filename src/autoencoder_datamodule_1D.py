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

class AutoEncoderDatamodule_1D(pl.LightningDataModule):
    
    def __init__(self, input_da, dl_kw, norm_stats, manage_nan: str = "suppress", 
                 profile_ratio: float = None,  data_selection: str = "random", dtype_str = 'float32'):
        super().__init__()
        self.input = input_da
        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        self.manage_nan = manage_nan
        self.profile_ratio = profile_ratio  # ratio for selecting profiles or time points
        self.data_selection = data_selection  # "random" or "spatial_sampling"
        self.space_ratio = 0.2        # For spatial_sampling mode (e.g. 0.5 means every 2 points)
        self.seed = 42                      # Fixed seed for reproducibility
        self.dtype_str = dtype_str
        self.coords = input_da.coords
        self.depth_array = self.coords["z"].data
        self.input_shape = input_da.data.shape
        self.depth_pre_treatment=dict(method="None", norm_on="None")
        self.train_da = None
        self.val_da = None
        self.test_da = None
        self.drop_last_batch = False
        self.is_data_normed = False
        # Set default for norm location if not specified; "AE" means normalization in AE_Dense.
        if "norm_location" not in self.norm_stats:
            self.norm_stats["norm_location"] = "AE"
        #self.generator = torch.Generator().manual_seed(self.seed)

    def setup(self, stage):
        required_dtype = getattr(np, self.dtype_str)
        if self.input.dtype != required_dtype:
            self.input = self.input.astype(required_dtype)

        if not self.is_data_normed:
            if self.manage_nan == "suppress":   ###TODO: add quadratic spline sliding window interpolation
                self.input = self.input.dropna(dim="lat")
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
            
            elif self.manage_nan == "before_normalization":
                self.input = self.input.fillna(0)

        # Data selection modification
        if self.data_selection == "random":
            # Stack all spatial dimensions into "profiles"
            da_stacked = self.input.stack(profiles=("time", "lat", "lon"))
            total_profiles = da_stacked.sizes["profiles"]
            # Compute number of profiles to select based on profile_ratio if provided
            if self.profile_ratio is not None and self.profile_ratio < 1:
                n_profiles = int(self.profile_ratio * total_profiles)
            else:
                n_profiles = total_profiles
            # Random permutation of indices with fixed seed and selection of n_profiles
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(total_profiles)[:n_profiles]
            da_stacked = da_stacked.isel(profiles=indices)
        elif self.data_selection == "spatial_sampling":
            # Spatial subsampling: select every kth point based on space_ratio
            if self.space_ratio is None or self.space_ratio <= 0 or self.space_ratio > 1:
                raise ValueError("space_ratio must be in (0,1] for spatial_sampling.")
            k = int(round(1 / self.space_ratio))
            # Select every kth latitude and longitude
            da_spatial = self.input.isel(lat=slice(0, None, k), lon=slice(0, None, k))
            # Randomly select a fraction of time points based on profile_ratio
            time_size = da_spatial.sizes["time"]
            if self.profile_ratio is not None and self.profile_ratio < 1:
                n_time = min(int(self.profile_ratio/self.space_ratio**2 * time_size), time_size)
            else:
                n_time = time_size
            rng = np.random.default_rng(self.seed)
            time_indices = rng.permutation(time_size)[:n_time]
            da_spatial = da_spatial.isel(time=sorted(time_indices))
            # Stack the remaining spatial dimensions into "profiles"
            da_stacked = da_spatial.stack(profiles=("time", "lat", "lon"))
        else:
            # Default: stack without reordering
            da_stacked = self.input.stack(profiles=("time", "lat", "lon"))
        
        # Continue with the rest of the code using da_stacked:
        total_profiles = da_stacked.sizes["profiles"]
        # Determine split sizes as before
        n_train = int(0.7 * total_profiles)
        n_val = int(0.2 * total_profiles)
        n_test = total_profiles - (n_train + n_val)
        # Split along profiles dimension
        train_da = da_stacked.isel(profiles=slice(0, n_train))
        val_da = da_stacked.isel(profiles=slice(n_train, n_train+n_val))
        test_da = da_stacked.isel(profiles=slice(n_train+n_val, total_profiles))

        if any(param is None for param in self.norm_stats['params'].values()):
            self.get_train_norm_stats(train_da.data)

        data = da_stacked.data

        # Only perform normalization in the datamodule if requested.
        if self.norm_stats["norm_location"] == "datamodule":
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
        # Otherwise leave data unnormalized for AE_Dense to handle.
        da_stacked.data = data

        train_da = da_stacked.isel(profiles=slice(0, n_train))
        val_da = da_stacked.isel(profiles=slice(n_train, n_train+n_val))
        test_da = da_stacked.isel(profiles=slice(n_train+n_val, total_profiles))

        # # Store the original multi-index (time, lat, lon) for each profile as a list of tuples in attributes.
        # self.train_coords = list(train_da.coords["profiles"].to_index())
        # self.val_coords = list(val_da.coords["profiles"].to_index())
        # self.test_coords = list(test_da.coords["profiles"].to_index())

        # Ensure the final DataArrays have dims: profiles, z.
        # After stacking, dims are ('z', 'profiles'), so we transpose.
        self.train_da = train_da.transpose("profiles", "z")
        self.val_da = val_da.transpose("profiles", "z")
        self.test_da = test_da.transpose("profiles", "z")

        self.min_val = self.input.data.min()
        self.max_val = self.input.data.max()

        self.is_data_normed = True

        if self.manage_nan == "after normalization":
            raise RuntimeError("a debugger, gérer pca + min")
            self.input = self.input.fillna(-6)


        if stage == 'fit':

            self.train_ds = AE_BaseDataset_3D(self.train_da)
            self.val_ds = AE_BaseDataset_3D(self.val_da)

        if stage == 'test':
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
        self.input = ipt

    def __len__(self):
        return len(self.input.profiles)
    
    def __getitem__(self, index):
        return self.input[index].data





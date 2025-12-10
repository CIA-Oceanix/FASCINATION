import pytorch_lightning as pl
import xarray as xr
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple
from sklearn.decomposition import PCA
import torch
import os
import pandas as pd
import hydra
from FASCINATION.src.utils import get_cfg_from_ckpt_path
from src.model.autoencoder.AE_CNN import AE_CNN


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

    def __init__(self, input_da, dl_kw, norm_stats, pooled_dim:str=None, depth_pre_treatment: dict={}, manage_nan: str = "supress_with_max_depth", n_profiles: int = None, reshape=None, rgb={"use": False, "method":None}, add_season=(False,False), dtype_str = 'float32'):
        super().__init__()
        self.input = input_da
        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        self.rgb = rgb

        #self.add_season,self.add_season_embedded = add_season
        self.pooled_dim = pooled_dim
        self.reshape = [] if reshape is None else reshape

        if self.rgb["use"]:
            self.norm_stats["method"] = "min_max"
            #self.add_season, self.add_season_embedded = False, False

        self.manage_nan = manage_nan
        self.n_profiles = n_profiles
        self.space_ratio = 0.2        # For spatial_sampling mode (e.g. 0.5 means every 2 points)
        self.time_ratio = 0.1
        self.depth_pre_treatment = depth_pre_treatment
        if pooled_dim != "spatial" or self.depth_pre_treatment["method"] != "pca":
            for key in self.depth_pre_treatment:
                self.depth_pre_treatment[key] = None

        self.dtype_str = dtype_str
        self.coords = input_da.coords
        self.depth_array = self.coords["z"].data
        self.input_shape = input_da.shape
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.drop_last_batch = False
        self.is_data_normed = False
        #self.generator = torch.Generator().manual_seed(42)
        self.seed = 42

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

            
            time_size, lat_size, lon_size = len(self.input.time), len(self.input.lat), len(self.input.lon)

            if self.n_profiles is not None:
                #n_times = max(self.n_profiles // (len(self.input.lat) * len(self.input.lon)), 10)
                while True:
                    space_factor = max(1, int(round(1 / self.space_ratio)))
                    time_factor = max(1,int(time_size * np.ceil(lat_size / space_factor) * np.ceil(lon_size / space_factor)) // self.n_profiles)
                    if time_factor <= time_size/10:
                        break
                    self.space_ratio *= 0.5
            else:
                time_factor, space_factor = 1, 1

            self.input = self.input.isel(time=slice(0,None,time_factor),lat=slice(0, None, space_factor), lon=slice(0, None, space_factor))
 
            if "factor_64" in self.reshape:
                lat_size = len(self.input.lat)
                lon_size = len(self.input.lon)
                n_lat = int(np.ceil(lat_size / 64))
                closest_lat_size = n_lat * 64
                n_lon = int(np.ceil(lon_size / 64))
                closest_lon_size = n_lon * 64
                new_lat = np.linspace(self.input.lat.min().item(), self.input.lat.max().item(), closest_lat_size)
                new_lon = np.linspace(self.input.lon.min().item(), self.input.lon.max().item(), closest_lon_size)
                self.input = self.input.interp(lat=new_lat, lon=new_lon, method="cubic")

            self.input.attrs["original_data"] = self.input.copy()

            if self.rgb["use"]:

                if self.rgb["method"] == "depth_layers":
                    data = self.input.data
                    n_channels = 3
                    z_dim = data.shape[1]
                    n_rgb = z_dim // n_channels
                    if n_rgb * n_channels != z_dim:
                        data = data[:, :n_rgb * n_channels, :, :]
                    data = data.reshape(data.shape[0], n_rgb, n_channels, data.shape[2], data.shape[3])
                    data = data.reshape(-1, n_channels, data.shape[3], data.shape[4])

                    new_time = pd.RangeIndex(data.shape[0], name="time")
                    new_z = np.arange(1, n_channels+1)
                    self.input = xr.DataArray(
                        data=data,
                        dims=("time", "z", "lat", "lon"),
                        coords={
                            "time": new_time,
                            "z": new_z,
                            "lat": self.input.lat,
                            "lon": self.input.lon
                        },
                        attrs=self.input.attrs
                    )
                    self.coords = self.input.coords
                    self.depth_array = self.coords["z"].data
                
                elif self.rgb["method"] == "CAE":
                    # Load pretrained CAE model to get RGB channels
                    cae_ckpt_path = "/Odyssey/private/o23gauvr/code/FASCINATION/outputs/remote/outputs/AE_CNN/pred_1_grad_0_max_pos_0_max_value_0_fft_0_weighted_1_inflection_pos_0_inflection_value_0/depth_pre_treatment_None_n_components_10_norm_on_components_train_on_components/dense_True/pooling_None_on_dim_dense/channels_[5000, 3000, 1000, 3]/upsample_mode_trilinear/linear_layer_False/cr_100000/1_conv_per_layer/padding_cubic/interp_size_5/final_upsample_upsample/act_fn_LeakyRelu/use_final_act_fn_True/lr_0.001/normalization_mean_std_along_depth/manage_nan_supress_with_max_depth/n_profiles_None/2025-10-21_20-41/checkpoints/best_checkpoint_loss.pth.tar"
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    cae_model = load_model(cae_ckpt_path, input_shape=self.input.shape, dl_kw=self.dl_kw, device=device, dtype_str= getattr(torch, self.dtype_str))

                    data = self.input.data
                    input_shape = data.shape

                    if cae_model.norm_stats["method"] == "min_max":
                        x_min = cae_model.norm_stats["params"]["x_min"]
                        x_max = cae_model.norm_stats["params"]["x_max"]
                        data = (data - x_min) / (x_max - x_min)
                    elif cae_model.norm_stats["method"] == "mean_std":
                        mean = cae_model.norm_stats["params"]["mean"]
                        std = cae_model.norm_stats["params"]["std"]
                        data = (data - mean) / std
                    elif cae_model.norm_stats["method"] == "mean_std_along_depth":
                        mean = cae_model.norm_stats["params"]["mean"]
                        std = cae_model.norm_stats["params"]["std"]
                        data = (data - mean) / std
                    #data_reshaped = data.transpose(0, 2, 3, 1).reshape(-1, input_shape[1])
                    with torch.no_grad():
                        data_tensor = torch.from_numpy(data).to(device=device, dtype=getattr(torch, self.dtype_str))
                        #cae_model.model_AE(data_tensor)

                        data_tensor = data_tensor.unsqueeze(-1)
                        batch_size = 10  # or any value that fits GPU memory
                        encoded_list = []
                        for i in range(0, len(data_tensor), batch_size):
                            batch = data_tensor[i:i + batch_size]
                            encoded = cae_model.model_AE.encoder(batch)
                            encoded_list.append(encoded.cpu())  # move to CPU if needed to free GPU memory

                        rgb_tensor = torch.cat(encoded_list, dim=0)
                        rgb_data = rgb_tensor.squeeze(-1).numpy()
                        del rgb_tensor
                        del data_tensor
                        del encoded_list
                        del encoded
                        torch.cuda.empty_cache()

                    self.input = xr.DataArray(
                        data=rgb_data,
                        dims=("time", "z", "lat", "lon"),
                        coords={
                            "time": self.input.time,
                            "z": [1, 2, 3],
                            "lat": self.input.lat,
                            "lon": self.input.lon
                        },
                        attrs=self.input.attrs
                    )
                    self.input.attrs['rgb_model'] = cae_model
                    self.coords = self.input.coords
                    self.depth_array = self.coords["z"].data

        # Split train, val, test



            n_times = len(self.input.time)

            rng = np.random.default_rng(self.seed)
            time_indices = rng.permutation(n_times)
            train_size = int(0.7 * n_times)
            val_size = int(0.2 * n_times)



            self.train_time_idx = time_indices[:train_size]
            self.val_time_idx = time_indices[train_size:train_size + val_size]
            self.test_time_idx = time_indices[train_size + val_size:]

            train_data_array = self.input.isel(time=self.train_time_idx).data

            if self.depth_pre_treatment.get("method") == "pca":
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
                if 'params' not in self.norm_stats or self.norm_stats['params'] is None or any(param is None for param in self.norm_stats['params'].values()):
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

            if self.depth_pre_treatment.get("method") == "pca":
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


            if self.depth_pre_treatment.get("method") == "pca":
                self.drop_last_batch = True

            self.is_data_normed = True

            if self.manage_nan == "after normalization":
                raise RuntimeError("a debugger, gérer pca + min")
                self.input = self.input.fillna(-6)


            # if self.add_season and not self.add_season_embedded:
            #     season_idx = xr.DataArray(
            #         [month_to_season(m) for m in pd.DatetimeIndex(self.input["time"]).month.values],
            #         coords={"time": self.input["time"]},
            #         dims=("time",),
            #         name="season_idx"
            #     )

            #     season_onehot = np.eye(4)[season_idx.values] 

            #     season_da = xr.DataArray(
            #         season_onehot[:, :, None, None] * np.ones((1, 1, len(self.input['lat']), len(self.input['lon']))),
            #         coords={
            #         "time": self.input["time"],
            #         "z": ["winter", "spring", "summer", "fall"],
            #         "lat": self.input['lat'],
            #         "lon": self.input['lon']
            #         },
            #         dims=("time", "z", "lat", "lon"),
            #         name="season_onehot",
            #     )   
                
            #     self.input = xr.concat([season_da.astype(self.input.dtype), self.input], dim="z")


            # elif self.add_season_embedded:
            #     season_idx = np.array([month_to_season(m) for m in pd.DatetimeIndex(self.input["time"]).month.values] )[:,None,None,None]*np.ones((1, 1, len(self.input['lat']), len(self.input['lon'])))
            #     #self.input = xr.concat([season_da.astype(self.input.dtype), self.input], dim="z")
            #     season_da = xr.DataArray(
            #         season_idx,
            #         coords={
            #             "time": self.input["time"],
            #             "z": ["season_idx"],
            #             "lat": self.input['lat'],
            #             "lon": self.input['lon']
            #         },
            #         dims=("time", "z", "lat", "lon"),
            #         name="season_onehot",
            # ) 

            #     self.input = xr.concat([season_da.astype(self.input.dtype), self.input], dim="z")


        required_dtype = getattr(np, self.dtype_str)
        if self.input.dtype != required_dtype:
            self.input = self.input.astype(required_dtype)

        assert self.input.dtype == required_dtype, f"input data type {self.input.dtype} does not match required dtype {required_dtype}"

        if stage == 'fit':
            self.train_da = self.input.isel(time=self.train_time_idx)
            self.train_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(self.train_da["time"]).month.values]
            self.train_shape = self.train_da.shape
            val_data_da = self.input.isel(time=self.val_time_idx)
            self.train_ds = AE_BaseDataset_3D(self.train_da)
            self.val_ds = AE_BaseDataset_3D(val_data_da)

        if stage == 'test':
            self.test_da = self.input.isel(time=self.test_time_idx)
            self.test_shape = self.test_da.shape
            self.test_da.attrs['season_idx'] = [month_to_season(m) for m in pd.DatetimeIndex(self.test_da["time"]).month.values]
            self.test_da.attrs['original_data'] = self.input.attrs["original_data"].sel(time=self.test_da["time"])
            if self.rgb["use"] and self.rgb["method"] == "CAE":
                self.test_da.attrs['rgb_model'] = self.input.attrs['rgb_model']
            self.test_ds = AE_BaseDataset_3D(self.test_da)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, drop_last=self.drop_last_batch, **self.dl_kw)

    def get_train_norm_stats(self, train_arr: np.array, verbose=False):
        self.norm_stats["params"] = {}
        if self.norm_stats['method'] == "mean_std":
            self.norm_stats["params"]["mean"] = np.nanmean(train_arr)
            self.norm_stats["params"]["std"] = np.nanstd(train_arr)
        elif self.norm_stats['method'] == "mean_std_along_depth":
            if self.depth_pre_treatment.get("norm_on") == "components":
                self.norm_stats["params"]["mean"] = np.nanmean(train_arr, axis=0).reshape(1, -1, 1, 1)
                self.norm_stats["params"]["std"] = np.nanstd(train_arr, axis=0).reshape(1, -1, 1, 1)
            else:
                self.norm_stats["params"]["mean"] = np.nanmean(train_arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
                self.norm_stats["params"]["std"] = np.nanstd(train_arr, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        elif self.norm_stats['method'] == "min_max":
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




def load_model(ckpt_path,input_shape=None,dl_kw=None, device='cpu', dtype_str='float32'):
    cfg = get_cfg_from_ckpt_path(ckpt_path)
    lit_mod = hydra.utils.call(cfg.model)
    batch_shape = (dl_kw['batch_size'],*input_shape[1:]) 
    lit_mod.model_hparams["input_shape"] = batch_shape
    lit_mod.model_AE = AE_CNN(**lit_mod.model_hparams)
    lit_mod.set_last_activation_function()
    lit_mod.model_AE.to(device=device, dtype=dtype_str)

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    checkpoint["state_dict"] = checkpoint["state_dict"]()  ##TODO supress this line after fixing saving issue
    lit_mod.load_state_dict(checkpoint["state_dict"],strict=False)
    lit_mod.norm_stats = checkpoint["norm_stats"]
    lit_mod = lit_mod.eval()

    for param in lit_mod.parameters():
        param.requires_grad = False

    return lit_mod
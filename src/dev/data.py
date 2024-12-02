import pytorch_lightning as pl
import xarray as xr
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple
import functools as ft
import pickle 
import os
from typing import Union,TypeVar
TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class AutoEncoderDatamodule_1D(pl.LightningDataModule):
    
    def __init__(self, input_da, dl_kw, norm_stats, manage_nan: str = "suppress", n_profiles: int = None,  dtype_str = 'float32',):
        super().__init__()
        self.input = input_da
        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        
        self.manage_nan = manage_nan
        self.n_profiles = n_profiles
        
        self.dtype_str = dtype_str


        self.test_time = None
        self.test_var = None
        self.test_lat = None
        self.test_lon = None
        self.test_z = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False
        
        self.generator = torch.Generator().manual_seed(42)

    
    def setup(self, stage):

        
        if self.manage_nan == "suppress":
            self.input = self.input.dropna(dim="lat")
        
        elif self.manage_nan == "before_normalization":
            self.input = self.input.fillna(0)
            
        
        
        if self.n_profiles is not None:
            

            time_indices = torch.randint(0, len(self.input.time), (self.n_profiles,), generator=self.generator)
            lat_indices = torch.randint(0, len(self.input.lat), (self.n_profiles,), generator=self.generator)
            lon_indices =torch.randint(0, len(self.input.lon), (self.n_profiles,), generator=self.generator)

            # # Step 3: Create arrays of selected coordinates
            selected_times = self.input.time[time_indices].values
            selected_lats = self.input.lat[lat_indices].values
            selected_lons = self.input.lon[lon_indices].values

            # Step 4: Split the selected coordinates into train, validation, and test sets
            train_size = int(0.7 * self.n_profiles)
            val_size = int(0.2 * self.n_profiles)


            # Generate indices for splitting
            indices = torch.randperm(self.n_profiles, generator=self.generator)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Step 5: Use these indices to extract train, validation, and test sets
            train_times = selected_times[train_indices]
            train_lats = selected_lats[train_indices]
            train_lons = selected_lons[train_indices]

            val_times = selected_times[val_indices]
            val_lats = selected_lats[val_indices]
            val_lons = selected_lons[val_indices]

            test_times = selected_times[test_indices]
            test_lats = selected_lats[test_indices]
            test_lons = selected_lons[test_indices]

            train_data_array = self.input.sel(
                time=xr.DataArray(train_times, dims="profiles"),
                lat=xr.DataArray(train_lats, dims="profiles"),
                lon=xr.DataArray(train_lons, dims="profiles")
                )
            

        if not self.is_data_normed:
            
            if any(param is None for param in self.norm_stats['params'].values()):
                self.get_train_norm_stats(train_data_array)
            # input_train, target_train = self.input.isel(time=train_da.indices), self.target.isel(time=train_da.indices)
            # mean, std = self.norm_stats(input_train, target_train)
            if self.norm_stats["method"] == "min_max":
                x_min = self.norm_stats["params"]["x_min"]
                x_max = self.norm_stats["params"]["x_max"]
                self.input = (self.input - x_min)/(x_max - x_min) 
                
            if self.norm_stats["method"] == "mean_std":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                self.input = (self.input - mean)/(std) 
                
            if self.norm_stats["method"] == "mean_std_along_depth":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                self.input = (self.input - mean)/(std) 
                
            #self.target["cutoff_freq"] = (self.target["cutoff_freq"])/10000  
            #self.target["ecs"] = (self.target["ecs"])/670.25141631
            self.is_data_normed = True
                 
            if self.manage_nan == "after normalization":      
                self.input = self.input.fillna(-6)  #? -6
                


            
        
        if stage == 'fit':


            train_data_array = self.input.sel(
                time=xr.DataArray(train_times, dims="profiles"),
                lat=xr.DataArray(train_lats, dims="profiles"),
                lon=xr.DataArray(train_lons, dims="profiles")
                )
                

            val_data_array = self.input.sel(
                time=xr.DataArray(val_times, dims="profiles"),
                lat=xr.DataArray(val_lats, dims="profiles"),
                lon=xr.DataArray(val_lons, dims="profiles")
                )          

            
            self.train_ds = AE_BaseDataset_1D(
                train_data_array, dtype_str = self.dtype_str 
                )  #self.target.isel(time=train_time_idx.indices)
            self.val_ds = AE_BaseDataset_1D(
                val_data_array, dtype_str= self.dtype_str
                )
            
            
        if stage == 'test':
            
            test_data_array = self.input.sel(
                time=xr.DataArray(test_times, dims="profiles"),
                lat=xr.DataArray(test_lats, dims="profiles"),
                lon=xr.DataArray(test_lons, dims="profiles")
                )
            # self.val_ds = BaseDataset(
            #     self.input.isel(time=val_da.indices), self.target.isel(time=val_da.indices)
            # )
            self.test_ds = AE_BaseDataset_1D(
                test_data_array, dtype_str = self.dtype_str
            )
            
            self.test_time = self.test_ds.input["time"]
            #self.test_var = self.test_ds.tgt["variable"]
            self.test_lat = self.test_ds.input["lat"]
            self.test_lon = self.test_ds.input["lon"]
            self.test_z = self.test_ds.input["z"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    
    
    def get_train_norm_stats(self,train_da, verbose = False):
        if self.norm_stats.method == "mean_std":
            self.norm_stats["params"]["mean"] = train_da.mean(skipna = True).item()
            self.norm_stats["params"]["std"] = train_da.std(skipna = True).item()

        
        elif self.norm_stats.method == "mean_std_along_depth":
            self.norm_stats["params"]["mean"] = train_da.mean(skipna = True, dim = ("time","lat",'lon'))
            self.norm_stats["params"]["std"] = train_da.std(skipna = True, dim = ("time","lat",'lon'))


        elif self.norm_stats.method == "min_max":
            self.norm_stats["params"]["x_min"] = train_da.min(skipna = True).item()
            self.norm_stats["params"]["x_max"] = train_da.max(skipna = True).item()
            

        if verbose:
            print("Norm stats", self.norm_stats)

            
        return self.norm_stats
    
    

class AE_BaseDataset_1D(torch.utils.data.Dataset):
    def __init__(self, ipt, dtype_str = 'float32'):
        super().__init__()
        
        self.input = ipt

        self.dtype = getattr(np, dtype_str)
        
    def __len__(self):
        return len(self.input.profiles)
    
    def __getitem__(self, index):
        return self.input[index].data.astype(self.dtype) 






class AutoEncoderDatamodule_2D(pl.LightningDataModule):
    
    def __init__(self, input_da, dl_kw, norm_stats, manage_nan: str = "suppress", n_profiles: int = None,  dtype_str = 'float32',):
        super().__init__()
        self.input = input_da
        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        
        self.manage_nan = manage_nan
        self.n_profiles = n_profiles
        
        self.dtype_str = dtype_str


        self.test_time = None
        self.test_var = None
        self.test_lat = None
        self.test_lon = None
        self.test_z = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False
        
        self.generator = torch.Generator().manual_seed(42)

    
    def setup(self, stage):

        
        if self.manage_nan == "suppress":
            self.input = self.input.dropna(dim="lat")
        
        elif self.manage_nan == "before_normalization":
            self.input = self.input.fillna(0)
            
        
        
        if self.n_profiles is not None:
            
            n_times = self.n_profiles/(len(self.input.lat)*len(self.input.lon))

            time_indices = torch.randint(0, len(self.input.time), (n_times,), generator=self.generator)


            # # Step 3: Create arrays of selected coordinates
            selected_times = self.input.time[time_indices].values


            # Step 4: Split the selected coordinates into train, validation, and test sets
            train_size = int(0.7 * n_times)
            val_size = int(0.2 * n_times)


            # Generate indices for splitting
            indices = torch.randperm(n_times, generator=self.generator)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Step 5: Use these indices to extract train, validation, and test sets
            train_times = selected_times[train_indices]
            val_times = selected_times[val_indices]
            test_times = selected_times[test_indices]

            train_data_array = self.input.sel(time=train_times)
            

        if not self.is_data_normed:
            
            if any(param is None for param in self.norm_stats['params'].values()):
                self.get_train_norm_stats(train_data_array)
            # input_train, target_train = self.input.isel(time=train_da.indices), self.target.isel(time=train_da.indices)
            # mean, std = self.norm_stats(input_train, target_train)
            if self.norm_stats["method"] == "min_max":
                x_min = self.norm_stats["params"]["x_min"]
                x_max = self.norm_stats["params"]["x_max"]
                self.input = (self.input - x_min)/(x_max - x_min) 
                
            if self.norm_stats["method"] == "mean_std":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                self.input = (self.input - mean)/(std) 
                
            if self.norm_stats["method"] == "mean_std_along_depth":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                self.input = (self.input - mean)/(std) 
                
            #self.target["cutoff_freq"] = (self.target["cutoff_freq"])/10000  
            #self.target["ecs"] = (self.target["ecs"])/670.25141631
            self.is_data_normed = True
                 
            if self.manage_nan == "after normalization":      
                self.input = self.input.fillna(-6)  #? -6
                

        test_data_array = self.input.sel(time=test_times)
        

        
        if stage == 'fit':

        
            train_data_array = self.input.sel(time=train_times)
            val_data_array = self.input.sel(time=val_times)

            self.train_ds = AE_BaseDataset_2D(
                    train_data_array, dtype_str= self.dtype_str
                    )
                    
            self.val_ds = AE_BaseDataset_2D(
                    val_data_array, dtype_str= self.dtype_str
                    )
        
        if stage == 'test':

            self.test_ds = AE_BaseDataset_1D(
                test_data_array, dtype_str = self.dtype_str
            )
            
            
            self.test_time = self.test_ds.input["time"]
            self.test_lat = self.test_ds.input["lat"]
            self.test_lon = self.test_ds.input["lon"]
            self.test_z = self.test_ds.input["z"]
            

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    
    
    def get_train_norm_stats(self,train_da, verbose = False):
        if self.norm_stats.method == "mean_std":
            self.norm_stats["params"]["mean"] = train_da.mean(skipna = True).item()
            self.norm_stats["params"]["std"] = train_da.std(skipna = True).item()

        
        elif self.norm_stats.method == "mean_std_along_depth":
            self.norm_stats["params"]["mean"] = train_da.mean(skipna = True, dim = ("time","lat",'lon'))
            self.norm_stats["params"]["std"] = train_da.std(skipna = True, dim = ("time","lat",'lon'))


        elif self.norm_stats.method == "min_max":
            self.norm_stats["params"]["x_min"] = train_da.min(skipna = True).item()
            self.norm_stats["params"]["x_max"] = train_da.max(skipna = True).item()
            

        if verbose:
            print("Norm stats", self.norm_stats)

            
        return self.norm_stats
    
    

class AE_BaseDataset_2D(torch.utils.data.Dataset):
    def __init__(self, ipt, dtype_str = 'float32'):
        super().__init__()
        
        self.input, self.tgt = ipt.transpose('time', 'z', 'lat', 'lon')

        self.dtype = getattr(np, dtype_str)
        
    def __len__(self):
        return len(self.input.time)
    
    def __getitem__(self, index):
        return self.input[index].data.astype(self.dtype) 


    
    
    
    
class BaseDatamodule(pl.LightningDataModule):
    def __init__(self, input_da, dl_kw, norm_stats, dim: str ="2D", manage_nan: str = "suppress", n_profiles: int = None,  dtype_str = 'float32',):
        super().__init__()
        self.input = input_da[0]
        self.target = input_da[1]
        self.dl_kw = dl_kw
        self.dim = dim
        self.norm_stats = norm_stats
        
        self.manage_nan = manage_nan
        self.n_profiles = n_profiles
        
        self.dtype_str = dtype_str


        self.test_time = None
        self.test_var = None
        self.test_lat = None
        self.test_lon = None
        self.test_z = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False
        
        self.generator = torch.Generator().manual_seed(42)

    
    def setup(self, stage):

        
        if self.manage_nan == "suppress":
            self.input = self.input.dropna(dim="lat")
        
        elif self.manage_nan == "before_normalization":
            self.input = self.input.fillna(0)
            
        
        
        if self.n_profiles is not None:
            

            time_indices = torch.randint(0, len(self.input.time), (self.n_profiles,), generator=self.generator)
            lat_indices = torch.randint(0, len(self.input.lat), (self.n_profiles,), generator=self.generator)
            lon_indices =torch.randint(0, len(self.input.lon), (self.n_profiles,), generator=self.generator)

            # # Step 3: Create arrays of selected coordinates
            selected_times = self.input.time[time_indices].values
            selected_lats = self.input.lat[lat_indices].values
            selected_lons = self.input.lon[lon_indices].values

            # Step 4: Split the selected coordinates into train, validation, and test sets
            train_size = int(0.7 * self.n_profiles)
            val_size = int(0.2 * self.n_profiles)
            test_size = self.n_profiles - train_size - val_size

            # Generate indices for splitting
            indices = torch.randperm(self.n_profiles, generator=self.generator)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Step 5: Use these indices to extract train, validation, and test sets
            train_times = selected_times[train_indices]
            train_lats = selected_lats[train_indices]
            train_lons = selected_lons[train_indices]

            val_times = selected_times[val_indices]
            val_lats = selected_lats[val_indices]
            val_lons = selected_lons[val_indices]

            test_times = selected_times[test_indices]
            test_lats = selected_lats[test_indices]
            test_lons = selected_lons[test_indices]

            train_data_array = self.input.sel(
                time=xr.DataArray(train_times, dims="profiles"),
                lat=xr.DataArray(train_lats, dims="profiles"),
                lon=xr.DataArray(train_lons, dims="profiles")
                )
            

        if not self.is_data_normed:
            
            if any(param is None for param in self.norm_stats['params'].values()):
                self.get_train_norm_stats(train_data_array)
            # input_train, target_train = self.input.isel(time=train_da.indices), self.target.isel(time=train_da.indices)
            # mean, std = self.norm_stats(input_train, target_train)
            if self.norm_stats["method"] == "min_max":
                x_min = self.norm_stats["params"]["x_min"]
                x_max = self.norm_stats["params"]["x_max"]
                self.input = (self.input - x_min)/(x_max - x_min) 
                
            if self.norm_stats["method"] == "mean_std":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                self.input = (self.input - mean)/(std) 
                
            if self.norm_stats["method"] == "mean_std_along_depth":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                self.input = (self.input - mean)/(std) 
                
            #self.target["cutoff_freq"] = (self.target["cutoff_freq"])/10000  
            #self.target["ecs"] = (self.target["ecs"])/670.25141631
            self.target = self.target/670.25141631   ##TODO: manage tgt
            self.is_data_normed = True
                 
            if self.manage_nan == "after normalization":      
                self.input = self.input.fillna(-6)  #? -6
                

        train_data_array = self.input.sel(
            time=xr.DataArray(train_times, dims="profiles"),
            lat=xr.DataArray(train_lats, dims="profiles"),
            lon=xr.DataArray(train_lons, dims="profiles")
            )
            
        test_data_array = self.input.sel(
            time=xr.DataArray(test_times, dims="profiles"),
            lat=xr.DataArray(test_lats, dims="profiles"),
            lon=xr.DataArray(test_lons, dims="profiles")
            )
        
        val_data_array = self.input.sel(
            time=xr.DataArray(val_times, dims="profiles"),
            lat=xr.DataArray(val_lats, dims="profiles"),
            lon=xr.DataArray(val_lons, dims="profiles")
            )
            
        
        if stage == 'fit':
            self.train_ds = BaseDataset(
                train_data_array, None, dim = self.dim, dtype_str = self.dtype_str 
                )  #self.target.isel(time=train_time_idx.indices)
            self.val_ds = BaseDataset(
                val_data_array, None, dim = self.dim, dtype_str= self.dtype_str
                )
        if stage == 'test':
            # self.val_ds = BaseDataset(
            #     self.input.isel(time=val_da.indices), self.target.isel(time=val_da.indices)
            # )
            self.test_ds = BaseDataset(
                test_data_array, None, dim=self.dim,  dtype_str = self.dtype_str
            )
            self.test_time = self.test_ds.input["time"]
            #self.test_var = self.test_ds.tgt["variable"]
            self.test_lat = self.test_ds.input["lat"]
            self.test_lon = self.test_ds.input["lon"]
            self.test_z = self.test_ds.input["z"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    
    
    def get_train_norm_stats(self,train_da, verbose = False):
        if self.norm_stats.method == "mean_std":
            self.norm_stats["params"]["mean"] = train_da.mean(skipna = True).item()
            self.norm_stats["params"]["std"] = train_da.std(skipna = True).item()

        
        elif self.norm_stats.method == "mean_std_along_depth":
            self.norm_stats["params"]["mean"] = train_da.mean(skipna = True, dim = ("time","lat",'lon'))
            self.norm_stats["params"]["std"] = train_da.std(skipna = True, dim = ("time","lat",'lon'))


        elif self.norm_stats.method == "min_max":
            self.norm_stats["params"]["x_min"] = train_da.min(skipna = True).item()
            self.norm_stats["params"]["x_max"] = train_da.max(skipna = True).item()
            

        if verbose:
            print("Norm stats", self.norm_stats)

            
        return self.norm_stats
    
    

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, ipt, tgt, dim: str = "1D", dtype_str = 'float32'):
        super().__init__()
        
        if dim =="1D":
            #self.input, self.tgt  = ipt.stack(merged_dim=('time', 'lat', 'lon')).transpose("merged_dim","z"), tgt.stack(merged_dim=('time', 'lat', 'lon'))
            self.input = ipt
            self.tgt = ipt #tgt
        else:
            self.input, self.tgt = ipt.transpose('time', 'z', 'lat', 'lon'), tgt.transpose('time', 'lat', 'lon')   #tgt.to_array().transpose('time', 'variable', 'lat', 'lon')
        self.dtype = getattr(np, dtype_str)
        
    def __len__(self):
        return len(self.input.profiles)
        #return min(len(self.input.time), len(self.tgt.time))
    
    def __getitem__(self, index):
        return TrainingItem._make((self.input[index].data.astype(self.dtype), self.tgt[index].data.astype(self.dtype)))  ###? Do we need a precision of float64 if we want our normalize ECS to be precise to the mm ?

        ##TODO put the nan_to_num higher in the chain, in load ssf for exemple
    
    
    
    
    
class BaseDatamodule_ecs_classif(pl.LightningDataModule):
    def __init__(self, input_da, dl_kw, model_name: str, x_min = 1459.0439165829073, x_max = 1552.54994512, dtype_str = 'float32' ):
        super().__init__()
        self.input = input_da[0]
        self.target = input_da[1]
        self.dl_kw = dl_kw
        
        self.x_min = x_min
        self.x_max = x_max
        
        self.dtype_str = dtype_str
        self.model_name = model_name


        self.test_time = None
        self.test_var = None
        self.test_lat = None
        self.test_lon = None
        self.test_z = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False
        

    
    def setup(self, stage):
        train_time_idx, val_time_idx, test_time_idx = torch.utils.data.random_split(self.input.time.data,
                                                                                    [0.7, 0.2, 0.1], 
                                                                                    generator=torch.Generator().manual_seed(42))
        if not self.is_data_normed:

            if self.x_max is None or self.x_min is None:
                self.x_min, self.x_max = np.nanmin(self.input.celerity.values), np.nanmax(self.input.celerity.values)
            self.input = (self.input - self.x_min)/(self.x_max - self.x_min) # min max normalization, hard coded values for now because it saves computation time

            self.is_data_normed = True
        
        if stage == 'fit':
            
            if self.model_name == "UNet_3D":
                self.train_ds = BaseDataset_ecs_classif_3D(
                    self.input.isel(time=train_time_idx.indices), self.target.isel(time=train_time_idx.indices), dtype_str = self.dtype_str
                    )
                self.val_ds = BaseDataset_ecs_classif_3D(
                    self.input.isel(time=val_time_idx.indices), self.target.isel(time=val_time_idx.indices), dtype_str= self.dtype_str
                    )
                
            else:
                
                self.train_ds = BaseDataset_ecs_classif_2D(
                    self.input.isel(time=train_time_idx.indices), self.target.isel(time=train_time_idx.indices), dtype_str = self.dtype_str
                    )
                self.val_ds = BaseDataset_ecs_classif_2D(
                    self.input.isel(time=val_time_idx.indices), self.target.isel(time=val_time_idx.indices), dtype_str= self.dtype_str
                    )
                
        if stage == 'test':
            if self.model_name == "UNet_3D":
                self.test_ds = BaseDataset_ecs_classif_3D(
                    self.input.isel(time=test_time_idx.indices), self.target.isel(time=test_time_idx.indices), dtype_str = self.dtype_str
                )

                self.test_z = self.test_ds.input["z"]
                
            else:
                self.test_ds = BaseDataset_ecs_classif_2D(
                    self.input.isel(time=test_time_idx.indices), self.target.isel(time=test_time_idx.indices), dtype_str = self.dtype_str
                )

            self.test_time = self.test_ds.tgt["time"]
            self.test_lat = self.test_ds.tgt["lat"]
            self.test_lon = self.test_ds.tgt["lon"]
            
            

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    
    def norm_stats(self, ipt, target):
        mean, std = {}, {}
        mean["input"] = ipt.mean()
        std["input"] = ipt.std()
        for j in target.data_vars:
            mean[j] = target[j].mean()
            std[j] = target[j].std()

        return mean, std

class BaseDataset_ecs_classif_3D(torch.utils.data.Dataset):
    def __init__(self, ipt, tgt, dtype_str = 'float32'):
        super().__init__()
        self.input, self.tgt = ipt.transpose('time', 'z', 'lat', 'lon'), tgt.transpose('time','z', 'lat', 'lon')   #tgt.to_array().transpose('time', 'variable', 'lat', 'lon')
        self.dtype = getattr(np, dtype_str)
        
    def __len__(self):
        return min(len(self.input.time), len(self.tgt.time))
    
    def __getitem__(self, index):
        return TrainingItem._make((np.nan_to_num(self.input[index].data.astype(self.dtype)), self.tgt[index].data.astype(self.dtype)))   ###? Do we need a precision of float64 if we want our normalize ECS to be precise to the mm ?
    
    
    
    
    
    
    
    
class BaseDataset_ecs_classif_2D(torch.utils.data.Dataset):
    def __init__(self, ipt, tgt, dtype_str = 'float32'):
        super().__init__()
        self.input, self.tgt = ipt.transpose('time', 'z', 'lat', 'lon'), tgt.transpose('time', 'lat', 'lon')   #tgt.to_array().transpose('time', 'variable', 'lat', 'lon')
        self.dtype = getattr(np, dtype_str)
        
    def __len__(self):
        return min(len(self.input.time), len(self.tgt.time))
    
    def __getitem__(self, index):
        return TrainingItem._make((np.nan_to_num(self.input[index].data.astype(self.dtype)), self.tgt[index].data.astype(self.dtype)))   ###? Do we need a precision of float64 if we want our normalize ECS to be precise to the mm ?
# class BaseDataModule(pl.LightningDataModule):
#     def __init__(self, input_da, domains, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, **kwargs):
#         super().__init__()
#         self.input_da = input_da
#         self.domains = domains
#         self.xrds_kw = xrds_kw
#         self.dl_kw = dl_kw
#         self.aug_kw = aug_kw if aug_kw is not None else {}
#         self._norm_stats = norm_stats

#         self.train_ds = None
#         self.val_ds = None
#         self.test_ds = None
#         self._post_fn = None

#     def norm_stats(self):
#         if self._norm_stats is None:
#             self._norm_stats = self.train_mean_std()
#             print("Norm stats", self._norm_stats)
#         return self._norm_stats

#     def train_mean_std(self, variable='tgt'):
#         train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])
#         return train_data.sel(variable=variable).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))

#     def post_fn(self):
#         m, s = self.norm_stats()
#         normalize = lambda item: (item - m) / s
#         return ft.partial(ft.reduce,lambda i, f: f(i), [
#             TrainingItem._make,
#             lambda item: item._replace(tgt=normalize(item.tgt)),
#             lambda item: item._replace(input=normalize(item.input)),
#         ])


#     def setup(self, stage='test'):
#         train_data = self.input_da.sel(self.domains['train'])
#         post_fn = self.post_fn()
#         self.train_ds = XrDataset(
#             train_data, **self.xrds_kw, postpro_fn=post_fn,
#         )
#         if self.aug_kw:
#             self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

#         self.val_ds = XrDataset(
#             self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn,
#         )
#         self.test_ds = XrDataset(
#             self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,
#         )


#     def train_dataloader(self):
#         return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

# class ConcatDataModule(BaseDataModule):
#     def train_mean_std(self):
#         sum, count = 0, 0
#         train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {}))
#         for domain in self.domains['train']:
#             _sum, _count = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: (da.sum(), da.pipe(np.isfinite).sum()))
#             sum += _sum
#             count += _count

#         mean = sum / count
#         sum = 0
#         for domain in self.domains['train']:
#             _sum = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: da - mean).pipe(np.square).sum()
#             sum += _sum
#         std = (sum / count)**0.5
#         return mean.values.item(), std.values.item()

#     def setup(self, stage='test'):
#         post_fn = self.post_fn()
#         self.train_ds = XrConcatDataset([
#             XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
#             for domain in self.domains['train']
#         ])
#         if self.aug_factor >= 1:
#             self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

#         self.val_ds = XrConcatDataset([
#             XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
#             for domain in self.domains['val']
#         ])
#         self.test_ds = XrConcatDataset([
#             XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
#             for domain in self.domains['test']
#         ])


# class RandValDataModule(BaseDataModule):
#     def __init__(self, val_prop, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.val_prop = val_prop

#     def setup(self, stage='test'):
#         post_fn = self.post_fn()
#         train_ds = XrDataset(self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn,)
#         n_val = int(self.val_prop * len(train_ds))
#         n_train = len(train_ds) - n_val
#         self.train_ds, self.val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

#         if self.aug_factor > 1:
#             self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

#         self.test_ds = XrDataset(self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,)
    




# TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])


# class XrDataset(torch.utils.data.Dataset):
#     """
#     torch Dataset based on an xarray.DataArray with on the fly slicing.

#     ### Usage: #### 
#     If you want to be able to reconstruct the input

#     the input xr.DataArray should:
#         - have coordinates
#         - have the last dims correspond to the patch dims in same order
#         - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)

#     the batches passed to self.reconstruct should:
#         - have the last dims correspond to the patch dims in same order
#     """
#     def __init__(
#             self, da, patch_dims, domain_limits=None, strides=None,
#             check_full_scan=False, check_dim_order=False,
#             postpro_fn=None
#             ):
#         """
#         da: xarray.DataArray with patch dims at the end in the dim orders
#         patch_dims: dict of da dimensions to size of a patch 
#         domain_limits: dict of da dimensions to slices of domain to select for patch extractions
#         strides: dict of dims to stride size (default to one)
#         check_full_scan: Boolean: if True raise an error if the whole domain is not scanned by the patch size stride combination
#         """
#         super().__init__()
#         self.return_coords = False
#         self.postpro_fn = postpro_fn
#         self.da = da.sel(**(domain_limits or {}))
#         self.patch_dims = patch_dims
#         self.strides = strides or {}
#         da_dims = dict(zip(self.da.dims, self.da.shape))
#         self.ds_size = {
#             dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
#             for dim in patch_dims
#         }


#         if check_full_scan:
#             for dim in patch_dims:
#                 if (da_dims[dim] - self.patch_dims[dim]) % self.strides.get(dim, 1) != 0:
#                     raise IncompleteScanConfiguration(
#                         f"""
#                         Incomplete scan in dimensions dim {dim}:
#                         dataarray shape on this dim {da_dims[dim]}
#                         patch_size along this dim {self.patch_dims[dim]}
#                         stride along this dim {self.strides.get(dim, 1)}
#                         [shape - patch_size] should be divisible by stride
#                         """
#                     )

#         if check_dim_order:
#             for dim in patch_dims:
#                 if not '#'.join(da.dims).endswith('#'.join(list(patch_dims))): 
#                     raise DangerousDimOrdering(
#                         f"""
#                         input dataarray's dims should end with patch_dims 
#                         dataarray's dim {da.dims}:
#                         patch_dims {list(patch_dims)}
#                         """
#                 )
#     def __len__(self):
#         size = 1
#         for v in self.ds_size.values():
#             size *= v
#         return size

#     def __iter__(self):
#         for i in range(len(self)):
#             yield self[i]

#     def get_coords(self):
#         self.return_coords = True
#         coords = []
#         try:
#             for i in range(len(self)):
#                 coords.append(self[i])
#         finally:
#             self.return_coords = False
#             return coords

#     def __getitem__(self, item):
#         sl = {
#                 dim: slice(self.strides.get(dim, 1) * idx,
#                            self.strides.get(dim, 1) * idx + self.patch_dims[dim])
#                 for dim, idx in zip(self.ds_size.keys(),
#                                     np.unravel_index(item, tuple(self.ds_size.values())))
#                 }
#         item =  self.da.isel(**sl)

#         if self.return_coords:
#             return item.coords.to_dataset()[list(self.patch_dims)]

#         item = item.data.astype(np.float32)
#         if self.postpro_fn is not None:
#             return self.postpro_fn(item)
#         return item

#     def reconstruct(self, batches, weight=None):
#         """
#         takes as input a list of np.ndarray of dimensionss (b, *, *patch_dims)
#         return a stitched xarray.DataArray with the coords of patch_dims

#     batches: list of torch tensor correspondin to batches without shuffle
#         weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
#         overlapping patches will be averaged with weighting 
#         """

#         items = list(itertools.chain(*batches))
#         return self.reconstruct_from_items(items, weight)

#     def reconstruct_from_items(self, items, weight=None):
#         if weight is None:
#             weight = np.ones(list(self.patch_dims.values()))
#         w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

#         coords = self.get_coords()

#         new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
#         dims = new_dims + list(coords[0].dims)

#         das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
#                for  it, co in zip(items, coords)]

#         da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
#         new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

#         rec_da = xr.DataArray(
#                 np.zeros([*new_shape.values(), *da_shape.values()]),
#                 dims=dims,
#                 coords={d: self.da[d] for d in self.patch_dims} 
#         )
#         count_da = xr.zeros_like(rec_da)

#         for da in das:
#             rec_da.loc[da.coords] = rec_da.sel(da.coords) + da * w
#             count_da.loc[da.coords] = count_da.sel(da.coords) + w

#         return rec_da / count_da

# class XrConcatDataset(torch.utils.data.ConcatDataset):
#     """
#     Concatenation of XrDatasets
#     """
#     def reconstruct(self, batches, weight=None):
#         """
#         Returns list of xarray object, reconstructed from batches
#         """
#         items_iter = itertools.chain(*batches)
#         rec_das = []
#         for ds in self.datasets:
#             ds_items = list(itertools.islice(items_iter, len(ds)))
#             rec_das.append(ds.reconstruct_from_items(ds_items, weight))
    
#         return rec_das

# class AugmentedDataset(torch.utils.data.Dataset):
#     def __init__(self, inp_ds, aug_factor, aug_only=False, noise_sigma=None):
#         self.aug_factor = aug_factor
#         self.aug_only = aug_only
#         self.inp_ds = inp_ds
#         self.perm = np.random.permutation(len(self.inp_ds))
#         self.noise_sigma = noise_sigma

#     def __len__(self):
#         return len(self.inp_ds) * (1 + self.aug_factor - int(self.aug_only))

#     def __getitem__(self, idx):
#         if self.aug_only:
#             idx = idx + len(self.inp_ds)

#         if idx < len(self.inp_ds):
#             return self.inp_ds[idx]

#         tgt_idx = idx % len(self.inp_ds)
#         perm_idx = tgt_idx
#         for _ in range(idx // len(self.inp_ds)):
#             perm_idx = self.perm[perm_idx]
        
#         item = self.inp_ds[tgt_idx]
#         perm_item = self.inp_ds[perm_idx]

#         noise = np.zeros_like(item.input, dtype=np.float32)
#         if self.noise_sigma is not None:
#             noise = np.random.randn(*item.input.shape).astype(np.float32) * self.noise_sigma

#         return item._replace(input=noise + np.where(np.isfinite(perm_item.input),
#                              item.tgt, np.full_like(item.tgt,np.nan)))
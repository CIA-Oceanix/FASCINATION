import pytorch_lightning as pl
import xarray as xr
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple
from sklearn.decomposition import PCA
import functools as ft
import pickle 
import os
from typing import Union,TypeVar
TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class AutoEncoderDatamodule_1D(pl.LightningDataModule):
    
    def __init__(self, input_da, dl_kw, norm_stats, manage_nan: str = "suppress", n_profiles: int = None,  dtype_str = 'float32'):
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
        self.test_da = None
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




    


class AutoEncoderDatamodule_3D(pl.LightningDataModule):
    
    def __init__(self, input_da, dl_kw, norm_stats, manage_nan: str = "suppress", n_profiles: int = None, depth_pre_treatment: dict = None, dtype_str = 'float32'):
        super().__init__()
        self.input = input_da
        self.dl_kw = dl_kw
        self.norm_stats = norm_stats
        
        self.manage_nan = manage_nan
        self.n_profiles = n_profiles
        
        self.depth_pre_treatment = depth_pre_treatment
        
        self.dtype_str = dtype_str

        self.coords = input_da.coords
        self.depth_array = self.coords["z"].data
        self.input_shape = input_da.data.shape

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None


        self.is_data_normed = False
        
        self.generator = torch.Generator().manual_seed(42)

    
    def setup(self, stage):

        required_dtype = getattr(np,self.dtype_str)
        if self.input.dtype != required_dtype:
            self.input = self.input.astype(required_dtype)

        
        if not self.is_data_normed:

            if self.manage_nan == "suppress":
                self.input = self.input.dropna(dim="lat")
        
            elif self.manage_nan == "before_normalization":
                self.input = self.input.fillna(0)

                
            if self.n_profiles is not None:
                n_times = max(self.n_profiles//(len(self.input.lat)*len(self.input.lon)),10)

            else: 
                n_times = len(self.input.time)
        
        
            time_indices = torch.randint(0, len(self.input.time), (n_times,), generator=self.generator)

            # Step 4: Split the selected coordinates into train, validation, and test sets
            train_size = int(0.7 * n_times)
            val_size = int(0.2 * n_times)


            self.train_time_idx = time_indices[:train_size]
            self.val_time_idx = time_indices[train_size:train_size + val_size]
            self.test_time_idx = time_indices[train_size + val_size:]
                
            train_data_array = self.input.isel(time=self.train_time_idx).data

            self.train_shape = self.input.isel(time=self.train_time_idx).shape
            self.test_shape = self.input.isel(time=self.test_time_idx).shape

            if self.depth_pre_treatment["method"] == "pca":
                input_shape = self.input.shape
                n_components = self.depth_pre_treatment["params"]
                pca = PCA(n_components = n_components, svd_solver = 'auto')
                
                train_data_array = pca.fit_transform(train_data_array.transpose(0,2,3,1).reshape(-1, input_shape[1]))
                #train_data_array = train_data_array.reshape(train_shape[0], train_shape[2], train_shape[3],n_components).transpose(0,3,1,2)
                
                self.depth_pre_treatment["fitted_pca"] = pca

                # pca_input = pca.transform(self.input.data.transpose(0,2,3,1).reshape(-1,train_data_array.shape[1]))
                # pca_input = pca_input.reshape(self.input_shape[0],self.input_shape[2],self.input_shape[3],n_components).transpose(0,3,1,2)

                # self.input = xr.DataArray(data=pca_input,
                #                           coords = {"time":self.coords["time"],
                #                                     "z": np.arange(n_components),
                #                                     "lat": self.coords["lat"],
                #                                     "lon":self.coords["lon"]})
                
            
            if any(param is None for param in self.norm_stats['params'].values()):
                self.get_train_norm_stats(train_data_array)

                #self.get_train_norm_stats(train_arr=pca_reduced_train_arr)
            
            
            if self.depth_pre_treatment["method"] == "pca":
                data = pca.transform(self.input.data.transpose(0,2,3,1).reshape(-1,input_shape[1])).reshape(input_shape[0], input_shape[2], input_shape[3], n_components).transpose(0,3,1,2)
                #self.input.data = pca.inverse_transform(data).reshape(input_shape[0], input_shape[2], input_shape[3], input_shape[1]).transpose(0,3,1,2)
            
            else:
                data = self.input.data
            

            if self.norm_stats["method"] == "min_max":
                x_min = self.norm_stats["params"]["x_min"]
                x_max = self.norm_stats["params"]["x_max"]
                data = (data - x_min)/(x_max - x_min) 
                

            elif self.norm_stats["method"] == "mean_std":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                data = (data - mean)/std
                
            elif self.norm_stats["method"] == "mean_std_along_depth":
                mean = self.norm_stats["params"]["mean"]
                std = self.norm_stats["params"]["std"]
                data = (data - mean)/std
            

            if self.depth_pre_treatment["method"] == "pca":
                coords = self.input.coords
                self.input = xr.DataArray(data=data,
                                          coords = {"time":coords["time"],
                                                    "z":np.arange(1,n_components+1),
                                                    "lat":coords["lat"],
                                                    "lon":coords["lon"]})
                
                self.coords = self.input.coords
                #self.input.data = pca.inverse_transform(data).reshape(input_shape[0], input_shape[2], input_shape[3], input_sh ape[1]).transpose(0,3,1,2)
            else:
                self.input.data = data

            
            #self.target["cutoff_freq"] = (self.target["cutoff_freq"])/10000  
            #self.target["ecs"] = (self.target["ecs"])/670.25141631
            self.is_data_normed = True
                 
            if self.manage_nan == "after normalization":   
                raise RuntimeError("a debugger, gérer pca + min")   
                self.input = self.input.fillna(-6)  #? -6

        
        if stage == 'fit':
            
            self.train_data_da = self.input.isel(time=self.train_time_idx)
                
            val_data_da = self.input.isel(time=self.val_time_idx)

            self.train_ds = AE_BaseDataset_3D(
                    self.train_data_da)
                    
            self.val_ds = AE_BaseDataset_3D(
                    val_data_da)
            
        
        if stage == 'test':

            self.test_data_da = self.input.isel(time=self.test_time_idx)

            self.test_ds = AE_BaseDataset_3D(
                self.test_data_da
                )
            
            
            # self.test_time = self.test_ds.input["time"]
            # self.test_lat = self.test_ds.input["lat"]
            # self.test_lon = self.test_ds.input["lon"]
            # self.test_z = self.test_ds.input["z"]
            

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    

    def get_train_norm_stats(self, train_arr: np.array, verbose = False):
        
        #train_arr = self.input.isel(time=train_time_idx).data


        if self.norm_stats.method == "mean_std":
            self.norm_stats["params"]["mean"] = np.nanmean(train_arr)
            self.norm_stats["params"]["std"] =  np.nanstd(train_arr)
        

        elif self.norm_stats.method == "mean_std_along_depth":

            if self.depth_pre_treatment["method"] == "pca":
                self.norm_stats["params"]["mean"] = np.nanmean(train_arr, axis = 0).reshape(1,-1)
                self.norm_stats["params"]["std"] = np.nanstd(train_arr, axis = 0).reshape(1,-1)

            else:
                self.norm_stats["params"]["mean"] = np.nanmean(train_arr, axis = (0,2,3)).reshape(1,-1,1,1)
                self.norm_stats["params"]["std"] = np.nanstd(train_arr, axis = (0,2,3)).reshape(1,-1,1,1)



        elif self.norm_stats.method == "min_max":
            
            self.norm_stats["params"]["x_min"] = np.nanmin(train_arr)
            self.norm_stats["params"]["x_max"] =  np.nanmax(train_arr)
     

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


    
    
   
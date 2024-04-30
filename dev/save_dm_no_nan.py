import xarray as xr
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
#import xrft
import matplotlib.pyplot as plt
#import pandas as pd
from tqdm import tqdm
import os
import glob
from pathlib import Path
from typing import Union, List
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import torch.nn.functional as F
import cartopy.crs as ccrs
import pickle

from src.utils import load_ssf_acoustic_variables, load_sound_speed_fields
from src.data import TrainingItem, AutoEncoderDatamodule, BaseDatamodule
from src.acoustic_predictor import ConvBlock, AcousticPredictor
from src.autoencoder import AutoEncoder



sound_speed_path = "/DATASET/eNATL/eNATL60_BLB002_sound_speed_regrid_0_1000m.nc"
ecs_path = "/DATASET/eNATL/eNATL60_BLB002_cutoff_freq_regrid_0_1000m.nc"



def test_dataloader_normalization(dataloader):
    
    for batch_number, batch in enumerate(dataloader):
        
        errors = []
        if batch.input.min() < -1:
            errors.append("Minimum value of input is less than -1.")
        if batch.input.max() > 1:
            errors.append("Maximum value of input is greater than 1.")
        if batch.tgt[:,0,:,:].min() < -1:
            errors.append("Minimum value of target ECS is less than -1.")
        if batch.tgt[:,0,:,:].max() > 1:
            errors.append("Maximum value of target ECS is greater than 1.")
        if abs(batch.input.max() - batch.input.min()) > 2:
            errors.append("Absolute difference between input max and min is greater than 2.")
        if abs(batch.tgt[:,0,:,:].max() - batch.tgt[:,0,:,:].min()) > 2:
            errors.append("Absolute difference between target ECS max and min is greater than 2.")
        
        if errors:
            error_message = "Validation failed for batch {}: {}".format(batch_number, "\n ".join(errors))
            raise ValueError(error_message)
        
        
def load_datamodule(input_da: tuple,
                    x_min: Union[float, None] = None,
                    x_max: Union[float, None] = None,
                    phases: Union[str, List[str]] = ["train", "test"],
                    verbose:float = True
                    ):
    
    
    if isinstance(phases, str):
        phases = list(phases)
    
    elif isinstance(phases, list):
        pass
    
    else: 
        raise ValueError("Argument phases should be a string or a list of strings")
    
    dm_dict = dict()
    
    dl_kw = dict(batch_size = None,
                 num_workers = 1)
    
    fit_setup = False
    
    dm = BaseDatamodule(input_da, dl_kw = dl_kw, x_min=x_min, x_max =x_max)
    
    
    for phase in tqdm(phases, unit = "phase", desc = "Genereting Dataloaders on selected splits"):
        
        if verbose:
            print("phase:", phase)


        
        # print(input_da[0].celerity.data.min(),input_da[0].celerity.data.max())
        # #coords = input_da.isel(domains[phase]).coords


        if phase == 'train':

            if not fit_setup:
                if verbose:
                    print("\t Train setup")
                dm.setup(stage = 'fit')
                fit_setup = True
            
            if verbose:
                print("\tGenerating dataloader")
                
            dm.dl_kw['batch_size'] = len(dm.train_ds.volume.coords['time'])
            dataloader = dm.train_dataloader()
        
        
        elif phase == "val":

            
            if not fit_setup:
                if verbose:
                    print("\t Validation setup")
                dm.setup(stage = 'fit')
                fit_setup = True
            
            if verbose:
                print("\tGenerating dataloader")
            dm.dl_kw['batch_size'] = len(dm.val_ds.volume.coords['time'])
            dataloader = dm.val_dataloader()
        
        
        elif phase == 'test':  
            if verbose:
                print("\t Test setup")
            dm.setup(stage = 'test')
            
            if verbose:
                print("\tGenerating dataloader")
            
            dm.dl_kw['batch_size'] = len(dm.test_ds.volume.coords['time'])
            dataloader = dm.test_dataloader()

        else: 
            raise ValueError("phase in phases should be train, val or test")
        
        if len(dataloader) != 1:
                raise ValueError("Dataloader should only contain 1 batch. This should never be seen for debugging purpose only")

        if verbose:
            print("Testing dataloader normalization")
        test_dataloader_normalization(dataloader)


        batch = next(iter(dataloader))

        dm_dict[phase] = batch
        
        
    return dm_dict



if __name__ == '__main__':
    
    x_min = 1459.0439165829073
    x_max = 1545.8698054910844

    ss_ds, acc_ds = load_ssf_acoustic_variables(sound_speed_path,ecs_path)
    ss_ds = ss_ds.dropna(dim='lat')
    acc_ds = acc_ds.sel(lat = ss_ds.lat)
    input_da = (ss_ds,acc_ds)
    del ss_ds
    del acc_ds
    
    with open("/DATASET/envs/o23gauvr/tmp/input_da_no_nan.pkl", "wb") as file:
        pickle.dump(input_da, file)
        


    dm = load_datamodule(input_da,
                        x_min,
                        x_max
                        )
    
    with open("/DATASET/envs/o23gauvr/tmp/datamodule_test_train_no_nan.pkl", "wb") as file:
        pickle.dump(dm, file)
        
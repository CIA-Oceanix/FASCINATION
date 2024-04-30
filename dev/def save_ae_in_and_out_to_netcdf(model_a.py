import xarray as xr
import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd
from tqdm import tqdm
import os
import glob
from pathlib import Path
from typing import Union, List


from src.utils import load_ssf_acoustic_variables, load_sound_speed_fields
from src.data import TrainingItem, BaseDatamodule, BaseDataset, AutoEncoderDatamodule
from src.acoustic_predictor import ConvBlock, AcousticPredictor
from src.autoencoder import AutoEncoder


def save_ae_in_and_out_to_netcdf(model_ae_path: str,
                                 input_da_path: str = "/DATASET/eNATL/eNATL60_BLB002_sound_speed_regrid_0_1000m.nc",
                                 model_ap_path: str = "/homes/o23gauvr/Documents/thèse/code/FASCINATION/outputs/accoustic_predictor/",
                                 saving_path: str = "/DATASET/envs/o23gauvr/autoencoder_outputs/",
                                 x_min: Union[float, None] = None,
                                 x_max: Union[float, None] = None,
                                 phases: Union[str, List[str]] = ["train", "val", "test"],
                                 domains: dict = dict(
                                     train = dict(time = slice(0, 254, None)),
                                     val = dict(time = slice(254, 331, None)),
                                     test = dict(time = slice(331, 365, None))
                                     ),
                                 gpu: Union[int, None] = 0,
                                 verbose: bool = True
                                ):

    
    
    if torch.cuda.is_available() and gpu is not None:
    ##This may not be necessary outside the notebook
        dev = f"cuda:{gpu}"
    else:
        dev = "cpu"
    
    device = torch.device(dev)

    if verbose:
        print("Selected device:", device)
    

    stage = dict(train = "fit",
                val = "fit",
                test = "test"
                )

    ap_indicator, arch_shape =  model_ae_path.split("/")[-2:]
    model_ae_path = glob.glob(os.path.join(model_ae_path, "**/*.ckpt"), recursive=True)[0]
    if verbose:
        print("Selected AutoEncoder model:", model_ae_path)
    
    if ap_indicator == "AE_with_AP" : 
        model_ap_path = glob.glob(os.path.join(model_ap_path, "**/*.ckpt"), recursive=True)[0]
        ## By default the first chekpoint is selected
        
    else:
        model_ap_path = None
        
    if verbose:
        print("Selected Accoustic Predictor model:", str(model_ap_path)) 
        print("Loading celerity data")
        
    input_da = load_sound_speed_fields(input_da_path)  
    
    model_ae = AutoEncoder.load_from_checkpoint(model_ae_path,x_min = x_min, x_max = x_max, acoustic_predictor= model_ap_path, arch_shape = arch_shape).to(device)
    
    if x_min is None or x_max is None:
        if verbose:
            print("Finding celerity minimum and maximum for normalization")
        x_min, x_max = np.nanmin(input_da.celerity.values), np.nanmax(input_da.celerity.values)
    if verbose:
        print("Rounded minimum and maximum used for normalization\n", f"\t c_min: {np.round(x_min,2)}\n", f"\t c_max: {np.round(x_max,2)}")
    
    if isinstance(phases, str):
        phases = list(phases)
    
    elif isinstance(phases, list):
        pass
    
    else: 
        raise ValueError("Argument phases should be a string or a list of strings")
        
    
    saving_path = os.path.join(saving_path,f"{ap_indicator}/{arch_shape}")
    
    
    for phase in tqdm(phases, unit = "phase", desc = "Treating splited phases"):
        if verbose:
            print("phase:", phase)

        dl_kw = {
        'batch_size': domains[phase]['time'].stop - domains[phase]['time'].start, 
        'num_workers': 1
        }

        coords = input_da.isel(domains[phase]).coords
        if verbose:
            print("\tGenerating dataloader")
        
        if ap_indicator == "AE_with_AP":
            dm = BaseDatamodule(input_da, domains, dl_kw, x_min=x_min, x_max =x_max)
            ###TODO: Gérer cette partie
            
        else:
            dm = AutoEncoderDatamodule(input_da, domains, dl_kw, x_min=x_min, x_max =x_max)
            
        dm.setup(stage = stage[phase])

        if phase == "train":
            dataloader = dm.train_dataloader()
            
        elif phase == "val":
            dataloader = dm.val_dataloader()
            
        elif phase == "test":
            dataloader = dm.test_dataloader()
            
            
        if len(dataloader) != 1:
            raise ValueError("Dataloader should only contain 1 batch. This should never be seen for debugging purpose only")
        if verbose:
            print("\tEncoding/Decoding ")
        batch = next(iter(dataloader))
        x, _ = batch
        x = x.to(device)
        
        # if not torch.equal(x,y):
        #     raise ValueError("x and y should be equal")

        if x.device != model_ae.device:
            raise ValueError(f"Dataloader and model should be on the same device. Dataloader is on {x.device}. Model is on {model_ae.device}")    

        output = model_ae(x)
        
        if not x.shape == output.shape:
            raise ValueError("Input and Output of the AutoEncoder should be of same shape")
        
        x,output = x*(x_max - x_min) + x_min, output*(x_max - x_min) + x_min
        
        
        
        if verbose:
            print("\tSaving input and output of AutoEncoder to:", os.path.join(saving_path, phase))
        
        Path(os.path.join(saving_path,phase)).mkdir(parents=True, exist_ok=True)    
        
        input_nc = xr.DataArray(data =  x.permute(2,3,1,0).cpu().detach().numpy(), coords = coords, name = f"celerity input {ap_indicator} {arch_shape} {phase}")
        input_nc.to_netcdf(os.path.join(saving_path,f"{phase}/celerity_input_{ap_indicator}_{arch_shape}_{phase}.nc"))
        del input_nc
        del x
        
        output_nc = xr.DataArray(data =  output.permute(2,3,1,0).cpu().detach().numpy(), coords = coords, name = f"celerity output {ap_indicator} {arch_shape} {phase}")
        output_nc.to_netcdf(os.path.join(saving_path,f"{phase}/celerity_output_{ap_indicator}_{arch_shape}_{phase}.nc"))
        del output_nc
        del output   
        
        
    torch.cuda.empty_cache()
        
        
if __name__ == "__main__":
    #save_ae_in_and_out_to_netcdf("/homes/o23gauvr/Documents/thèse/code/FASCINATION/outputs/AE_without_AP/4_15", gpu=1, verbose = True)
    pass
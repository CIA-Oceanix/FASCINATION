
import xarray as xr
import os
import glob
import yaml


def load_ssf_ecs_da(ssf_da_path, ecs_da_path):
    ssf_da = xr.open_dataarray(ssf_da_path)
    ecs_da = xr.open_dataarray(ecs_da_path)

    return ssf_da, ecs_da


def load_ssf_ecs_classif_da(ssf_da_path, ecs_classif_da_path):
    ssf_da = xr.open_dataarray(ssf_da_path)
    ecs_da = xr.open_dataarray(ecs_classif_da_path)   #.transpose("time", "lat", "lon", "z")

    return ssf_da, ecs_da


def get_ap_config_file_path_from_ckpt_path(ap_ckpt_path):
    splited_path = ap_ckpt_path.split(os.sep)
    
    try:
        index = splited_path.index('checkpoints')
        # Join the components back into a path up to the index before 'checkpoints'
        path_to_config = os.sep.join(splited_path[:index])
    except ValueError:
        # If 'checkpoints' is not in the path, return the original path
        print("The given path does not match the expected structure")
        
    config_path = glob.glob(f"{path_to_config}/.hydra/config.yaml")[0]
    
    return config_path
    
    

def get_ap_arch_shape_from_ckpt(ap_ckpt_path):
        
    config_path = get_ap_config_file_path_from_ckpt_path(ap_ckpt_path)
    
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)
        
    return config['model']['arch_shape']

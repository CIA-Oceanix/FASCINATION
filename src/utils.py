
import xarray as xr
import os
import hydra
import yaml
from IPython.display import Markdown, display
from omegaconf import OmegaConf
import torch
import pickle
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig
import src.differentiable_fonc as DF
import src.activation_function as AF
import os 
import sys

running_path = "/Odyssey/private/o23gauvr/code/FASCINATION/"
os.chdir(running_path)
sys.path.insert(0,running_path)


def load_ssp_da(ssf_da_path):
    
    return xr.open_dataarray(ssf_da_path)


def load_ssf_ecs_da(ssf_da_path, ecs_da_path):
    ssf_da = xr.open_dataarray(ssf_da_path)
    ecs_da = xr.open_dataarray(ecs_da_path)

    return ssf_da, ecs_da


def load_ssf_ecs_classif_da(ssf_da_path, ecs_classif_da_path):
    ssf_da = xr.open_dataarray(ssf_da_path)
    ecs_da = xr.open_dataarray(ecs_classif_da_path)   #.transpose("time", "lat", "lon", "z")

    return ssf_da, ecs_da



def load_model(model_ckpt_path: str,
               dm,
               batch: torch.Tensor, 
               verbose: bool = True):
    
    #sys.path.append('/Odyssey/private/o23gauvr/code/FASCINATION/src')

    if "/homes/o23gauvr/Documents/thèse/code/FASCINATION/" in sys.path:
        sys.path.remove("/homes/o23gauvr/Documents/thèse/code/FASCINATION/")

    cfg = get_cfg_from_ckpt_path(model_ckpt_path, pprint = False)
    
    lit_mod = hydra.utils.call(cfg.model)

    lit_mod = model_setup(lit_mod, dm, batch)

    # torch.serialization.add_safe_globals([DictConfig])

    checkpoint = torch.load(model_ckpt_path, weights_only=False, map_location=batch.device)
    lit_mod.load_state_dict(checkpoint["state_dict"],strict=False)
    #lit_mod.load_state_dict(torch.load(model_ckpt_path, map_location=device)["state_dict"])

    lit_mod.verbose = verbose

    lit_mod = lit_mod.eval() # Model in eval mode
    for param in lit_mod.parameters():
        param.requires_grad = False  # Ensure no gradients are calculated for this model

    return lit_mod


def model_setup(lit_model,
                dm,
                batch):
    
    lit_model.depth_pre_treatment = dm.depth_pre_treatment
    lit_model.norm_stats = dm.norm_stats
    lit_model.depth_arr = dm.depth_array


    lit_model.model_AE = lit_model.initiate_model(lit_model.model_name, lit_model.model_hparams, batch)

    if lit_model.depth_pre_treatment["method"] == "pca":
        pca = dm.depth_pre_treatment["fitted_pca"]
        lit_model.dif_pca_4D = DF.Differentiable4dPCA(pca, batch_shape=batch.shape, device=batch.device, dtype=getattr(torch,dm.dtype_str))     
    
    return lit_model


def set_last_activation_fucntion(lit_model, dm):

    if lit_model.norm_stats["method"] == "min_max":
        lit_model.model_AE.decoder.net[-1] = torch.Sigmoid()
    
    elif lit_model.norm_stats["method"] == "mean_std" or lit_model.norm_stats["method"] == "mean_std_along_depth":


        scale = max(abs(dm.min_val), abs(dm.max_val))
        scale = scale + 0.1*scale
        lit_model.model_AE.decoder.net[-1] = AF.ScaledTanh(scale)  ##See with nn.Identity()
    
    return lit_model




def loading_datamodule_phase(dm, phase = "fit"):
    

    dm.setup(stage = phase) 
    
    if phase == "fit": 
        ssp_arr = dm.train_da.data 

    elif phase == "test":
        ssp_arr = dm.test_da.data

    #ssp_tens = torch.tensor(ssp_arr).float().to(device)
                  
    return ssp_arr, dm



def loading_datamodule(dm):
    
    dm.setup(stage="fit") 
    dm.setup(stage="test") 
        
    train_ssp_arr = dm.train_da.data
    #train_ssp_tens = torch.tensor(train_ssp_arr).float().to(device)
    
    test_ssp_arr = dm.test_da.data
    #test_ssp_tens = torch.tensor(test_ssp_arr).float().to(device)
                
    return train_ssp_arr, test_ssp_arr, dm




def get_depth_array(ssf_da_path):
    return xr.open_dataarray(ssf_da_path).z.data


def get_convo_init_weight_bias(init_params_pickle_path):
    
    with open(init_params_pickle_path, "rb") as file:
        init_dic = pickle.load(file)
        # init_weight = init_dic["weight"] 
        # init_bias = init_dic["bias"]
    
    return init_dic #(init_bias, init_weight)
    


def get_config_file_path_from_ckpt_path(ap_ckpt_path):
    splited_path = ap_ckpt_path.split(os.sep)
    
    try:
        index = splited_path.index('checkpoints')
        # Join the components back into a path up to the index before 'checkpoints'
        path_to_config = os.sep.join(splited_path[:index])
    except ValueError:
        # If 'checkpoints' is not in the path, return the original path
        print("The given path does not match the expected structure")
        
    config_path = list(Path(path_to_config).rglob('config.yaml'))[0]    
    #config_path = glob.glob(f"{path_to_config}/**/config.yaml", recursive=True, include_hidden=True)[0]
    
    return config_path
    

def get_ap_arch_shape_from_ckpt(ap_ckpt_path):
        
    config_path = get_config_file_path_from_ckpt_path(ap_ckpt_path)
    
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)
        
    return config['model']['arch_shape']



def get_cfg_from_ckpt_path(ckpt_path, pprint = False):
    
    cfg_path = get_config_file_path_from_ckpt_path(ckpt_path)
    cfg = OmegaConf.load(cfg_path)
    
    if pprint:
        display(Markdown("""```yaml\n\n""" +yaml.dump(OmegaConf.to_container(cfg), default_flow_style=None, indent=2)+"""\n\n```"""))
        
        
    return cfg




def unorm_ssp_arr_3D(ssp_arr:np.array, dm, verbose = False):

    if verbose:
        print(dm.norm_stats.method)

    if dm.depth_pre_treatment["norm_on"] == "components":
        pca = dm.depth_pre_treatment["fitted_pca"]
        ssp_shape = ssp_arr.shape
        ssp_arr = pca.transform(ssp_arr.transpose(0,2,3,1).reshape(-1,ssp_shape[1])).reshape(ssp_shape[0], ssp_shape[2], ssp_shape[3], pca.n_components).transpose(0,3,1,2)


    if dm.norm_stats.method == "mean_std_along_depth":
        mean,std = dm.norm_stats.params.values()
        ssp_unorm_arr = (ssp_arr*std) + mean
        
    elif dm.norm_stats.method == "mean_std":
        mean,std = dm.norm_stats.params.values()
        ssp_unorm_arr = ssp_arr*std + mean
    
    elif dm.norm_stats.method == "min_max":
        x_min,x_max = dm.norm_stats.params.values()
        ssp_unorm_arr =ssp_arr*(x_max-x_min) + x_min
    

    if dm.depth_pre_treatment["norm_on"] == "components":

        ssp_unorm_arr = pca.inverse_transform(ssp_unorm_arr.transpose(0,2,3,1).reshape(-1,pca.n_components)).reshape(ssp_shape[0], ssp_shape[2], ssp_shape[3], len(dm.depth_array)).transpose(0,3,1,2)

        
    return ssp_unorm_arr


def cosanneal_lr_adamw(self, lr, T_max, weight_decay=0.):
    opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay= weight_decay)
    return {
        'optimizer': opt,
        'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max,
        ),
    }
            
def check_differentiable(input, model, verbose=True, raise_error=False):


    torch.autograd.set_detect_anomaly(True)        

    for name, param in model.named_parameters():

        if param.numel() == 0:
            continue

        if not param.requires_grad:
            if raise_error:
                raise RuntimeError(f"No gradient required for {name}")
            elif verbose:
                print(f"No gradient required for {name}")

    # Forward pass with gradient tracking
    z = model(input.requires_grad_(True))
    
    # Handle case where model returns a tuple
    if isinstance(z, tuple):
        z = z[0]  # Select the first tensor from the tuple
    
    # Define a simple loss as the sum of the output tensor
    loss = z.sum()
 
    try:
        # Backward pass
        loss.backward(retain_graph=True)
        
    except RuntimeError as e:
        raise ValueError("Anomaly detected:", e)
    
    # Check gradients for each model parameter
    for name, param in model.named_parameters():
        
        if param.numel() == 0:
            continue

        if param.grad is None:
            if raise_error:
                raise RuntimeError(f"No gradient computed for {name}")
            elif verbose:
                print(f"No gradient computed for {name}")
        elif verbose:
            pass
            #print(f"Gradient computed for {name}: {param.grad}")
        
    # Check if gradient is computed for the input tensor
    if input.grad is None:
        if raise_error:
            raise RuntimeError("No gradient computed for the input tensor")
        elif verbose:
            print("No gradient computed for the input tensor")
    elif verbose: 
        pass
        #print(f"Gradient computed for the input tensor: {input.grad}")


def check_abnormal_grad(model, writter, verbose=True, raise_error=False):

    grad_min = 1e-12
    grad_max = 1
    torch.autograd.set_detect_anomaly(True)        

    """Logs gradients after each backward pass."""
    parameters = {**dict(model.named_parameters())}  ##? grad in input ??
    for name, param in parameters.items():
        if param.grad is not None:
            # Log gradient histograms
            writter.add_histogram(f"{name}_grad", param.grad, model.global_step)

            if abs(param.grad.mean()) < grad_min:
                if raise_error:
                    raise RuntimeError(f"{name}, mean gradient < {grad_min}")
                elif verbose:
                    print(f"{name}, mean gradient < {grad_min}")
                    

            elif abs(param.grad.mean()) > grad_max:
                if raise_error:
                    raise RuntimeError(f"{name}, mean gradient > {grad_max}")

                elif verbose:
                    print(f"{name}, mean gradient > {grad_max}")


            elif param.grad.mean() == torch.nan:
                if raise_error:
                    raise RuntimeError(f"{name}, gradient mean is nan")

                elif verbose:
                    print(f"{name}, gradient mean is nan")

        elif (param.grad is None) and (param.numel() > 0):
            writter.add_histogram(f"{name}_grad", torch.zeros(1), model.global_step)
            if raise_error:
                raise RuntimeError(f"{name}, gradient is not computed")

            elif verbose:
                print(f"{name}, gradient is not computed")




        # if input.grad.min() < grad_min:
        #     if verbose:
        #         print(f"input gradient < {grad_min}")
        #     elif raise_error:
        #         raise RuntimeError(f"input gradient < {grad_min}")

        # elif input.grad.max() > grad_max:
        #     if verbose:
        #         print(f"input gradient < {grad_max}")
        #     elif raise_error:
        #         raise RuntimeError(f"input gradient < {grad_max}")

        # elif input.grad.mean() == torch.nan:
        #     if verbose:
        #         print(f"input gradient mean is nan")
        #     elif raise_error:
        #         raise RuntimeError(f"input gradient mean is nan")

if __name__ == "__main__":

    import sys

    running_path = "/homes/o23gauvr/Documents/thèse/code/FASCINATION/"
    sys.path.insert(0,running_path)
    os.chdir(running_path)

    cfg_path = "config/xp/autoencoder_V2.yaml"
    cfg = OmegaConf.load(cfg_path)
    # Load config
    dm = hydra.utils.call(cfg.datamodule)

    # Load data using loading_datamodule_phase
    train_arr_1, dm = loading_datamodule_phase(dm, phase='fit') #np.array([0]), np.array([0]), dm #


    # Load data using loading_datamodule
    train_arr_2, test_arr_2, dm  = loading_datamodule(dm)

    # Compare datasets
    print("Loading datamodule phase:")
    print("Numpy")
    print("Shape:", train_arr_1.shape)
    print("Mean:", np.mean(train_arr_1))
    print("Std Dev:", np.std(train_arr_1))

    print("\nTrain Data Summary:")
    print("Shape:", train_arr_2.shape)
    print("Mean:", np.mean(train_arr_2))
    print("Std Dev:", np.std(train_arr_2))

    # Check for differences
    differences = np.sum(train_arr_1 != train_arr_2)
    print("\nNumber of differences between datasets:", differences)

    print()


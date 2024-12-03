


import hydra
import pickle
from src.utils import get_cfg_from_from_ckpt_path
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import scipy.ndimage



def load_model(model_ckpt_path: str,
               device: str):

    cfg = get_cfg_from_from_ckpt_path(model_ckpt_path, pprint = False)
    
    lit_mod = hydra.utils.call(cfg.model)

    lit_mod.load_state_dict(torch.load(model_ckpt_path)["state_dict"])


    lit_mod = lit_mod.to(device) # Move model to gpu for faster inference
    lit_mod = lit_mod.eval() # Model in eval mode
    for param in lit_mod.parameters():
        param.requires_grad = False  # Ensure no gradients are calculated for this model

    return lit_mod



def unorm_ssp(ssp_arr:np.array, norm_stats: dict, verbose = True):

    if verbose:
        print(norm_stats.method)

    if norm_stats.method == "mean_std_along_depth":
        mean,std = norm_stats.params.values()
        ssp_unorm_arr = (ssp_arr*std.data.reshape(1,1,1,-1)) + mean.data.reshape(1,1,1,-1) 
        
    elif norm_stats.method == "mean_std":
        mean,std = norm_stats.params.values()
        ssp_unorm_arr = ssp_arr*std + mean
    
    elif norm_stats.method == "min_max":
        x_min,x_max = norm_stats.params.values()
        ssp_unorm_arr =ssp_arr*(x_max-x_min) + x_min
        
    return ssp_unorm_arr




def explicit_ecs_3D(ssp: torch.tensor,
                    depth_tens: torch.tensor,
                    tau = 100):
    ssp = ssp.unsqueeze(1).nan_to_num()          
    kernel = torch.tensor([-1.0, 1.0]).float().view(1,1,2,1,1).to(ssp.device)
    derivative = F.conv3d(ssp, kernel, padding=(0,0,0))

    #sign = DF.differentiable_sign(derivative)

    sign = torch.sign(derivative) + F.tanh(tau * derivative) - F.tanh(tau * derivative).detach()
    #print("After torch.sign (sign):", sign.requires_grad, sign.grad_fn)


    sign_diff = F.conv3d(sign, kernel, padding=(1,0,0))
    sign_change = F.tanh(10*F.relu(-sign_diff))

    for pattern in ([1, 0, 1], [1, -1, 0, 0]):  
        n = len(pattern)
        kernel_matrix = torch.eye(n)
        element_match = 0
        for i in range(n):
            kernel_element = kernel_matrix[i,:].view(1,1,n,1,1).to(ssp.device)
            element_match = element_match + (F.conv3d(sign, kernel_element, padding=(0,0,0)) - pattern[i])**2

        pattern_recognition = F.pad( element_match, (0, 0, 0, 0, 1, (sign_change.shape[2]- element_match.shape[2]) - 1),value=1.)    
        mask_discontinuity = 1 - F.relu(pattern_recognition+1) * F.relu(1-pattern_recognition)

        sign_change = sign_change * mask_discontinuity


    mask = F.relu(2 - torch.cumsum(sign_change, dim=2))

    depth_array_tens = depth_tens[:mask.shape[2]].view(1,-1,1,1).to(ssp.device).type(sign_change.dtype)
    depth_array_tens[0,0,0,0] = 0.  ##TODO the true first z value is equal to 48cm. It may have to be considered that way
    ecs_pred = (sign_change * mask ).squeeze(dim=1)
    ecs_pred = (ecs_pred * depth_array_tens).max(dim=1).values 
    return ecs_pred



def loading_datamodule_phase(dm, device = "cpu", phase = "fit"):
    
    dm = hydra.utils.call(cfg.datamodule) 
    dm.setup(stage = phase) 
    
    if phase == "fit": 
        ssp_ds = dm.train_dataloader().dataset.input

    elif phase == "test":
        ssp_ds = dm.test_dataloader().dataset.input
      
        
    ssp_arr = ssp_ds.dropna(dim='lat').data
    ssp_tens = torch.tensor(ssp_arr).float().to(device)
    
    coords = ssp_ds.coords

                
    return ssp_tens, dm.norm_stats, coords



if __name__ == "__main__":


    ckpt_list = list(Path("/homes/o23gauvr/Documents/thèse/code/FASCINATION/outputs/AE/AE_CNN_3D").rglob('*.ckpt'))

    gpu = 3
    if torch.cuda.is_available() and gpu is not None:
    ##This may not be necessary outside the notebook
        dev = f"cuda:{gpu}"
    else:
        dev = "cpu"

    device = torch.device(dev)

    dm_dict = None
    
    rmse_dict_ae_ssp_per_componnents = {}
    rmse_dict_ae_ecs_per_components = {}
    
    rmse_dict_ae_filtered_ssp_per_componnents = {}
    rmse_dict_ae_filtered_ecs_per_components = {}


    for ckpt_path in tqdm(ckpt_list):
        
        ckpt_path = str(ckpt_path)
        cfg = get_cfg_from_from_ckpt_path(ckpt_path, pprint = False)
        
        if dm_dict == None or cfg.datamodule != dm_dict:

            print("Inititing datamodule; Generating train and test datasets")
            dm_dict = cfg.datamodule
            dm = hydra.utils.call(dm_dict)
            
            test_ssp_tens, norm_stats, coords = loading_datamodule_phase(dm, device, phase = "test")
            depth_array = coords["z"].data
            
    
        lit_model = load_model(ckpt_path, device)
        ssp_pred = lit_model(test_ssp_tens)


        bottleneck_shape = lit_model.encoder.net(test_ssp_tens.unsqueeze(1)).shape   
        n_components = bottleneck_shape[2]

        ecs_truth = explicit_ecs_3D(test_ssp_tens, torch.tensor(depth_array).float()).to(device)
        
        ecs_pred_ecs = explicit_ecs_3D(ssp_pred,torch.tensor(depth_array).float()).to(device)
        ecs_pred_ecs_max = torch.tensor(depth_array[ssp_pred.argmax(axis=1).detach().cpu().numpy()]).float().to(device)

        
        rmse_dict_ae_ssp_per_componnents[ckpt_path]= torch.sqrt(torch.mean((test_ssp_tens - ssp_pred)**2))
        
        rmse_dict_ae_ecs_per_components[ckpt_path]= {"explicit":torch.sqrt(torch.mean((ecs_truth - ecs_pred_ecs)**2)).item(),
                                                     "max": torch.sqrt(torch.mean((ecs_truth - ecs_pred_ecs_max)**2)).item()}
        



        median_kernel = 4
        ssp_3D_median = scipy.ndimage.median_filter(ssp_pred.detach().cpu().numpy(),size=(1,median_kernel,1,1), mode = "wrap")
        ecs_pred_ae_median = explicit_ecs_3D(torch.tensor(ssp_3D_median, device = device).float(), torch.tensor(depth_array).float()).squeeze()
        ecs_pred_ae_median_max = depth_array[ssp_3D_median.argmax(axis=1)]
        
        ssp_rmse = np.sqrt(np.mean((ssp_pred.detach().cpu().numpy()-ssp_3D_median)**2))
        ecs_rmse = torch.sqrt(torch.mean((ecs_truth-ecs_pred_ae_median)**2)).item()
        ecs_rmse_max = np.sqrt(np.mean((ecs_truth.detach().cpu().numpy()-ecs_pred_ae_median_max)**2))
                
        rmse_dict_ae_filtered_ssp_per_componnents[ckpt_path]= ssp_rmse

        rmse_dict_ae_filtered_ecs_per_components[ckpt_path]= {"explicit":ecs_rmse,
                                                              "max": ecs_rmse_max}
        
    

    with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/rmse_ssp_ae_on_dm_3D.pkl', 'wb') as file:
        pickle.dump(rmse_dict_ae_ssp_per_componnents, file)
        
    with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/rmse_ecs_ae_on_dm_3D.pkl', 'wb') as file:
        pickle.dump(rmse_dict_ae_ecs_per_components, file)
        
    with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/rmse_ssp_ae_filtered_on_dm_3D.pkl', 'wb') as file:
        pickle.dump(rmse_dict_ae_filtered_ssp_per_componnents, file)
        
    with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/rmse_ecs_ae_filtered_on_dm_3D.pkl', 'wb') as file:
        pickle.dump(rmse_dict_ae_filtered_ecs_per_components, file)
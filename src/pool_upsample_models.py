import sys

# Add the parent directory to sys.path
parent_dir = "/homes/o23gauvr/Documents/thèse/code/FASCINATION/"
sys.path.insert(0, parent_dir)

import torch
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import yaml
from IPython.display import Markdown, display
from omegaconf import OmegaConf
from hydra.utils import call, instantiate
from pytorch_lightning import Trainer

import torch.nn as nn
import src.differentiable_fonc as DF
from src.utils import *


class NoConvAE(nn.Module):
    def __init__(self, 
                 n_layers:int, 
                 pooling_dim:str = "spatial",
                 pooling_mode:str = "Avg"):


        super().__init__()

        self.upsample_mode = "trilinear"


        if pooling_dim == "all":
            pool_str = 2
            
        elif pooling_dim == "depth":
            pool_str = (2,1,1)

        elif pooling_dim == "spatial":
            pool_str = (1,2,2)
        
        elif pooling_dim == None:
            pool_str = 1




        pooling_dict = {"Avg": nn.AvgPool3d(kernel_size= 1,stride=pool_str, padding = 0),
                        "Max": nn.MaxPool3d(kernel_size=1, stride=pool_str, padding = 0),
                        "None": nn.Identity()}     
        
        upsample_dict = {"Avg": nn.Upsample(scale_factor = pool_str, mode = self.upsample_mode),
                        "Max":  nn.Upsample(scale_factor = pool_str, mode = self.upsample_mode),
                        "None": nn.Identity()}   
        
        pool_layer = pooling_dict[pooling_mode]
        upsample_layer = upsample_dict[pooling_mode]

            

        self.encoder = nn.Sequential(*[pool_layer for i in range(n_layers)])
    
        self.decoder = nn.Sequential(*[upsample_layer for i in range(n_layers-1)])

        self.decoder.append(nn.Upsample(size=None, mode=self.upsample_mode))

    
    def forward(self,x):

        self.decoder[-1].size = x.shape[-3:]

        x = x.unsqueeze(1)

        self.bottleneck = self.encoder(x)

        self.output = self.decoder(self.bottleneck).squeeze(1)
    

        return self.output

        



if __name__ == "__main__":

    unorm = True

    gpu = 2

    pca_list = np.arange(1,108) #[1, 10, 30, 50, 100, 107] #np.arange(1,108)  #[1, 10, 50, 100, 107] #np.arange(107,108) 

    ae_rmse_dict = {"SSP":{},
                    "ECS":{}}
    
    n_layers = 6

        
    if torch.cuda.is_available() and gpu is not None:
    ##This may not be necessary outside the notebook
        dev = f"cuda:{gpu}"
    else:
        dev = "cpu"

    device = torch.device(dev)




    cfg_path = "config/xp/autoencoder_V2.yaml"
    cfg = OmegaConf.load(cfg_path)
    display(Markdown("""```yaml\n\n""" +yaml.dump(OmegaConf.to_container(cfg), default_flow_style=None, indent=2)+"""\n\n```"""))

    #trainer = Trainer(inference_mode = True)

    dm_dict = cfg.datamodule
    dm = call(dm_dict) #hydra.utils.call(dm_dict)
    train_ssp_tens, test_ssp_tens, dm, coords = loading_datamodule(dm)


    if unorm:
        test_ssp_tens = torch.tensor(unorm_ssp_arr_3D(test_ssp_tens.cpu().numpy(), dm)).to(device)





    for n_components in tqdm(pca_list):

        input_shape = test_ssp_tens.shape
        pca = PCA(n_components = n_components, svd_solver = 'auto')
        pca.fit(train_ssp_tens.permute(0,2,3,1).reshape(-1,input_shape[1]))
        
        dif_pca = DF.Differentiable4dPCA(pca,input_shape,device, dtype=test_ssp_tens.dtype) #test_ssp_tens.shape

        pca_reduced_test_ssp_tens = dif_pca.transform(test_ssp_tens) #test_ssp_tens
        pca_unreduced_test_ssp_tens = dif_pca.inverse_transform(pca_reduced_test_ssp_tens)

        pca_test = pca.inverse_transform(pca.transform(test_ssp_tens.detach().cpu().numpy().transpose(0,2,3,1).reshape(-1,input_shape[1]))).reshape(input_shape[0],input_shape[2],input_shape[3],input_shape[1]).transpose(0,3,1,2)
        assert torch.allclose(torch.tensor(pca_test,dtype=test_ssp_tens.dtype, device=test_ssp_tens.device), pca_unreduced_test_ssp_tens, atol=1e-6), "PCA transformation is not consistent"
        
        for keys in ae_rmse_dict.keys():
            ae_rmse_dict[keys][f"Pool_upsample_pca_{n_components}"] = {}

        for n_layer in tqdm(range(n_layers)): 
            
            model_ae = NoConvAE(n_layer, pooling_dim="spatial", pooling_mode="Avg")
            

            unreduced_pred_ssp_tens = model_ae(test_ssp_tens)
            reduced_pred_ssp_tens = model_ae.bottleneck.squeeze(1)
            reduced_bottleneck_shape = reduced_pred_ssp_tens.shape


    
            dif_pca.original_shape = reduced_bottleneck_shape
            pca_reduced_pred_ssp_tens = dif_pca.transform(reduced_pred_ssp_tens)
            pca_reduced_bottleneck_shape = pca_reduced_pred_ssp_tens.shape
            pca_unreduced_pred_ssp_tens = dif_pca.inverse_transform(pca_reduced_pred_ssp_tens)
            pred_ssp_tens = model_ae.decoder(pca_unreduced_pred_ssp_tens.unsqueeze(1)).squeeze(1)   

                    


            max_ssp_truth_idx = torch.tensor(np.nanargmax(test_ssp_tens.cpu().numpy(), axis=1))
            ecs_truth = dm.depth_array[max_ssp_truth_idx]        

            max_ssp_pred_idx = torch.tensor(np.nanargmax(pred_ssp_tens.cpu().numpy(), axis=1))
            ecs_pred = dm.depth_array[max_ssp_pred_idx]
            
            ae_rmse_dict["SSP"][f"Pool_upsample_pca_{n_components}"][pca_reduced_bottleneck_shape] = torch.sqrt(torch.mean((test_ssp_tens - pred_ssp_tens) ** 2)).item()
            ae_rmse_dict["ECS"][f"Pool_upsample_pca_{n_components}"][pca_reduced_bottleneck_shape] = np.sqrt(np.mean((ecs_truth - ecs_pred) ** 2))


    with open(f'pickle/pooling_upsampling_pca_pre_treatment_rmse__all_components_norm_{not(unorm)}.pkl', 'wb') as f:
        pickle.dump(ae_rmse_dict, f)

import sys
import os

running_path = "/homes/o23gauvr/Documents/thèse/code/FASCINATION/"
sys.path.insert(0,running_path)
os.chdir(running_path)

import hydra
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from src.utils import *
from src.differentiable_fonc import Differentiable4dPCA


import matplotlib.pyplot as plt


class NoConvAE(nn.Module):
    def __init__(self, 
                 n:int, 
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

            

        self.encoder = nn.Sequential(*[pool_layer for i in range(n)])
    
        self.decoder = nn.Sequential(*[upsample_layer for i in range(n-1)])

        self.decoder.append(nn.Upsample(size=None, mode=self.upsample_mode))

    
    def forward(self,x):

        self.decoder[-1].size = x.shape[-3:]

        x = x.unsqueeze(1)

        self.bottleneck = self.encoder(x)

        self.output = self.decoder(self.bottleneck).squeeze(1)
    

        return self.output







if __name__ == "__main__":

    verbose = True

    unorm = False 
    
    cfg_path = "config/xp/autoencoder_V2.yaml"
    cfg = OmegaConf.load(cfg_path)

    cfg.dtype = "float32"

    n_layers = 11

    pca_algo_dif = False

    gpu = None
        
    if torch.cuda.is_available() and gpu is not None:
    ##This may not be necessary outside the notebook
        dev = f"cuda:{gpu}"
    else:
        dev = "cpu"

    device = torch.device(dev)


    print("Inititing datamodule; Generating train and test datasets")

    dm = hydra.utils.call(cfg.datamodule)


    train_ssp_arr, test_ssp_arr, dm = loading_datamodule(dm)

    test_ssp_tens = torch.tensor(test_ssp_arr, dtype=getattr(torch,cfg.dtype), device=device)
    # train_ssp_arr = train_ssp_tens.detach().cpu().numpy()
    # test_ssp_arr = test_ssp_tens.detach().cpu().numpy()

    # train_ssp_arr, _, dm = loading_datamodule_phase(dm,device,phase="fit")
    # test_ssp_arr, test_ssp_tens, dm = loading_datamodule_phase(dm,device,phase="test")




    depth_array = dm.depth_array


    ecs_truth_idx = np.argmax(test_ssp_arr,axis=1)
    ecs_truth = depth_array[ecs_truth_idx]
    
            

    ae_rmse_dict = {"SSP":{},
                    "ECS":{},
                    "bottleneck_shape":{}}
    
    for n in range(n_layers):
        ae_rmse_dict["SSP"][f"Pool_upsample_{n}_layers"] = {}
        ae_rmse_dict["ECS"][f"Pool_upsample_{n}_layers"] = {}
        ae_rmse_dict["bottleneck_shape"][f"Pool_upsample_{n}_layers"] = {}
    

    for n_components in tqdm(range(1,108), unit = "components", desc = "Computing PCA components", disable = not(verbose)):

        pca = PCA(n_components = n_components, svd_solver = 'auto')
        pca.fit(train_ssp_arr.transpose(0,2,3,1).reshape(-1,train_ssp_arr.shape[1]))
        #pca_dict[n_components] = pca
        
        if pca_algo_dif:
            dif_pca = Differentiable4dPCA(pca, batch_shape=test_ssp_tens.shape , device = test_ssp_tens.device, dtype=test_ssp_tens.dtype)
            pca_reduced_test_ssp_tens = dif_pca.transform(test_ssp_tens) 

        else:
            pca_reduced_test_ssp_tens = torch.tensor(pca.transform(test_ssp_arr.transpose(0,2,3,1).reshape(-1,test_ssp_arr.shape[1])).reshape(test_ssp_arr.shape[0],test_ssp_arr.shape[2],test_ssp_arr.shape[3],n_components).transpose(0,3,1,2), dtype=test_ssp_tens.dtype, device=test_ssp_tens.device)

        for n_layer in tqdm(range(n_layers),disable=not(verbose), unit = "layers", desc = "Computing AE layers"): 
            
            model_ae = NoConvAE(n_layer, pooling_dim="spatial", pooling_mode="Avg")

            pooled_upsampled_test_ssp_tens = model_ae(pca_reduced_test_ssp_tens)

            bottleneck_shape = model_ae.bottleneck.squeeze(1).shape
            ae_rmse_dict["bottleneck_shape"][f"Pool_upsample_{n_layer}_layers"] = bottleneck_shape

            if pca_algo_dif:
                pca_unreduced_test_ssp_tens = dif_pca.inverse_transform(pooled_upsampled_test_ssp_tens)
                pca_unreduced_test_ssp_arr = pca_unreduced_test_ssp_tens.detach().cpu().numpy()

            else:
                pca_unreduced_test_ssp_arr = pca.inverse_transform(pooled_upsampled_test_ssp_tens.detach().cpu().numpy().transpose(0,2,3,1).reshape(-1,n_components)).reshape(test_ssp_arr.shape[0],test_ssp_arr.shape[2],test_ssp_arr.shape[3],test_ssp_arr.shape[1]).transpose(0,3,1,2)
                    

            if unorm:
                ae_rmse_dict["SSP"][f"Pool_upsample_{n_layer}_layers"][n_components] = np.sqrt(np.mean((unorm_ssp_arr_3D(test_ssp_arr, dm) - unorm_ssp_arr_3D(pca_unreduced_test_ssp_arr, dm)) ** 2))
                #print("SSP RMSE: ",np.sqrt(np.mean((unorm_ssp_arr_3D(test_ssp_arr, dm) - unorm_ssp_arr_3D(pca_unreduced_test_ssp_arr, dm)) ** 2)))
            
            else:
                ae_rmse_dict["SSP"][f"Pool_upsample_{n_layer}_layers"][n_components] = np.sqrt(np.mean((test_ssp_arr - pca_unreduced_test_ssp_arr) ** 2))
                #print("SSP RMSE: ",np.sqrt(np.mean((test_ssp_arr - pca_unreduced_test_ssp_arr) ** 2)))
            
            ecs_pred_idx = np.argmax(pca_unreduced_test_ssp_arr,axis=1)
            ecs_pred = depth_array[ecs_pred_idx] 


            ae_rmse_dict["ECS"][f"Pool_upsample_{n_layer}_layers"][n_components] = np.sqrt(np.mean((ecs_truth - ecs_pred) ** 2))


            # plt.plot(pca_unreduced_test_ssp_arr[0,:,0,0],depth_array, label = "pred")
            # plt.plot(test_ssp_arr[0,:,0,0],depth_array, label = "truth")
            # plt.gca().invert_yaxis()
            # plt.legend()
            # plt.show()



        

        
    # with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/trained_pca_on_dm_3D_norm_{not(unorm)}.pkl', 'wb') as file:
    #     pickle.dump(pca_dict, file)
        
    # with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/rmse_ssp_pca_on_dm_3D_norm_{not(unorm)}.pkl', 'wb') as file:
    #     pickle.dump(rmse_dict_pca_ssp_per_componnents, file)

    if pca_algo_dif:
        pca_name = "dif_pca"
    else:
        pca_name = "sklearn_pca"

    with open(f'pickle/rmse_pca_all_components_with_pooling_upsampling_norm_{not(unorm)}_{pca_name}.pkl', 'wb') as f:
        pickle.dump(ae_rmse_dict, f)
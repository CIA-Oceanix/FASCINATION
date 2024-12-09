
import sys
sys.path.insert(0,"/homes/o23gauvr/Documents/thèse/code/FASCINATION/")


import hydra
import pickle
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from src.utils import *









if __name__ == "__main__":

    unorm = True 
    
    cfg_path = "config/xp/autoencoder_V2.yaml"
    cfg = OmegaConf.load(cfg_path)
                
    print("Inititing datamodule; Generating train and test datasets")

    dm = hydra.utils.call(cfg.datamodule)
    train_ssp_tens, test_ssp_tens, dm, coords = loading_datamodule(dm)
    
    train_ssp_arr = train_ssp_tens.detach().cpu().numpy()
    test_ssp_arr = test_ssp_tens.detach().cpu().numpy()
    
    depth_array = dm.depth_array


        
    input_test_shape = test_ssp_arr.shape
    
    if unorm:
        train_ssp_arr = unorm_ssp_arr_3D(train_ssp_arr, dm)
        test_ssp_arr = unorm_ssp_arr_3D(test_ssp_arr, dm)
                

    pca_dict = {}
    rmse_dict_pca_ssp_per_componnents = {}
    rmse_dict_pca_ecs_per_components = {}
    

    for n_components in tqdm(range(106,108), unit = "components", desc = "Computing PCA components"):
        pca = PCA(n_components = n_components, svd_solver = 'auto')
        pca.fit(train_ssp_arr.transpose(0,2,3,1).reshape(-1,107))
        pca_dict[n_components] = pca
        ssp_pca_test_arr =  pca.inverse_transform(pca.transform(test_ssp_arr.transpose(0,2,3,1).reshape(-1,input_test_shape[1]))).reshape(input_test_shape[0],input_test_shape[2],input_test_shape[3],-1).transpose(0,3,1,2)

        # ecs_truth = explicit_ecs_3D(torch.tensor(test_ssp_arr).float(),torch.tensor(depth_array).float()).to("cpu")
        # ecs_pred_pca = explicit_ecs_3D(torch.tensor(ssp_pca_test_arr).float(),torch.tensor(depth_array).float()).to("cpu")
        
        ecs_truth_idx = np.argmax(test_ssp_arr,axis=1)
        ecs_truth = depth_array[ecs_truth_idx]
             
        ecs_pca_idx = np.argmax(ssp_pca_test_arr,axis=1)
        ecs_pca = depth_array[ecs_pca_idx]


        rmse_dict_pca_ssp_per_componnents[n_components]= np.sqrt(np.mean((test_ssp_arr - ssp_pca_test_arr)**2))
        rmse_dict_pca_ecs_per_components[n_components]= np.sqrt(np.mean((ecs_truth - ecs_pca)**2))


        

        
    with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/trained_pca_on_dm_3D_norm_{not(unorm)}.pkl', 'wb') as file:
        pickle.dump(pca_dict, file)
        
    with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/rmse_ssp_pca_on_dm_3D_norm_{not(unorm)}.pkl', 'wb') as file:
        pickle.dump(rmse_dict_pca_ssp_per_componnents, file)
        
    with open(f'/homes/o23gauvr/Documents/thèse/code/FASCINATION/pickle/rmse_ecs_pca_on_dm_3D_norm_{not(unorm)}.pkl', 'wb') as file:
        pickle.dump(rmse_dict_pca_ecs_per_components, file)
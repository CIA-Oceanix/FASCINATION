import sys
import os

running_path = "/Odyssey/private/o23gauvr/code/FASCINATION/"
sys.path.insert(0,running_path)
os.chdir(running_path)

import numpy as np
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm
from src.utils import loading_datamodule_phase, unorm_ssp_arr_3D
import pickle


class NoConvAE(nn.Module):
    def __init__(self, 
                 n:int, 
                 pooling_dim:str = "spatial",
                 pooling_mode:str = "Avg"):


        super().__init__()  

        self.pooling_dim = pooling_dim
        self.upsample_mode = "trilinear"


        if pooling_dim == "all":
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

        if self.pooling_dim == "all":
            x = x.transpose(0,1).unsqueeze(-1).unsqueeze(-1)
        x = x.unsqueeze(1)

        
        self.decoder[-1].size = x.shape[2:]
        self.bottleneck = self.encoder(x)
    
        self.output = self.decoder(self.bottleneck).squeeze(1)
        
        if self.pooling_dim == "all":
            self.output = self.output.squeeze(-1).squeeze(-1)
            self.output = self.output.transpose(0,1)


        return self.output


def get_min_max_idx(arr,axs=1, pad=True):
    grad = np.diff(arr,axis=axs)
    grad_sign = np.sign(grad)
    min_max = np.diff(grad_sign,axis=axs) 
    min_max = np.abs(np.sign(min_max))
    if pad:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axs] = (1, 1)
        min_max = np.pad(min_max, pad_width, 'constant', constant_values=1)
    return min_max


def get_interpolation_points_idx(min_max_mask, axis=1, reduction_rank=1):
    # Get the indices where min/max are detected
    min_max_indices = np.where(min_max_mask == 1)
    
    # Extract points to keep
    reduced_indices = np.sort(np.unique(min_max_indices[axis]))
    
    # Create new indices with intermediate points
    if reduction_rank > 0:
        new_indices = []
        for i in range(len(reduced_indices) - 1):
            new_indices.append(reduced_indices[i])
            interp_points = np.linspace(reduced_indices[i], reduced_indices[i+1], reduction_rank+2, endpoint=False)[1:]
            new_indices.extend(interp_points.astype(int))
        new_indices.append(reduced_indices[-1])
        reduced_indices = np.unique(new_indices)
    

    
    return reduced_indices




def get_f1_score(min_max_idx_truth, min_max_idx_ae, axs=1, kernel_size=10):
    import numpy as np
    from scipy.ndimage import convolve


    # Define the kernel based on the shape of the truth array
    kernel_shape = [1] * min_max_idx_truth.ndim
    kernel_shape[axs] = kernel_size  # Set the size of the kernel along the specified axis
    kernel = np.ones(kernel_shape)

    # Expand the truth array with the kernel
    truth_expanded = convolve(min_max_idx_truth, kernel, mode='constant', cval=0.0)
    ae_expanded = convolve(min_max_idx_ae, kernel, mode='constant', cval=0.0)

    # Compute the true positives
    true_positives = (truth_expanded > 0) & (min_max_idx_ae > 0)
    num_true_positives = np.sum(true_positives, axis=axs)

    # Compute the false positives
    false_positives = (truth_expanded == 0) & (min_max_idx_ae > 0)
    num_false_positives = np.sum(false_positives, axis=axs)

    # Compute the false negatives
    false_negatives = (min_max_idx_truth > 0) & (ae_expanded == 0)
    num_false_negatives = np.sum(false_negatives, axis=axs)

    # Compute precision and recall while avoiding division by zero
    precision_den = num_true_positives + num_false_positives
    recall_den = num_true_positives + num_false_negatives

    precision_score = np.where(precision_den == 0, 0, num_true_positives / precision_den)
    recall_score = np.where(recall_den == 0, 0, num_true_positives / recall_den)

    # Compute f1_score and avoid division by zero when both precision and recall are 0
    sum_scores = precision_score + recall_score
    f1_score = np.where(sum_scores == 0, 0, 2 * (precision_score * recall_score) / sum_scores)

    return f1_score




if __name__ == "__main__":

    verbose = True
    xp="autoencoder_V2" #autoencoder_V2 #dense_ae
    pooling_dim = "spatial" if xp == "autoencoder_V2" else "all"

    max_reducation_range = 10 
    n_layers = 10
    gpu = 0
    
    cfg_path = f"config/xp/{xp}.yaml"
    cfg = OmegaConf.load(cfg_path)


    print("Inititing datamodule; Generating train and test datasets")

    dm = hydra.utils.call(cfg.datamodule)

    test_ssp_arr, dm = loading_datamodule_phase(dm)

    input_size = test_ssp_arr.size


    if  dm.norm_stats["norm_location"] == "datamodule":
        test_ssp_arr = unorm_ssp_arr_3D(test_ssp_arr, dm)


    depth_array = dm.depth_array

    ecs_truth_idx = np.argmax(test_ssp_arr,axis=1)
    ecs_truth = depth_array[ecs_truth_idx]

    min_max_idx_truth = get_min_max_idx(test_ssp_arr, pad=False)


    #test_ssp_tens = torch.tensor(test_ssp_arr).to(device)



    # depth_array = dm.depth_array

    # ecs_truth_idx = np.argmax(test_ssp_arr,axis=1)
    # ecs_truth = depth_array[ecs_truth_idx]
    

    rmse_dict = {"SSP":{},
                 "ECS":{},
                 "mean_error_n_min_max":{},
                 "F1_score":{},
                 "cr":{}}
    
    
    for n in range(n_layers):
        rmse_dict["SSP"][f"Pool_upsample_{n}_layers"] = {}
        rmse_dict["ECS"][f"Pool_upsample_{n}_layers"] = {}
        rmse_dict["mean_error_n_min_max"][f"Pool_upsample_{n}_layers"] = {}
        rmse_dict["F1_score"][f"Pool_upsample_{n}_layers"] = {}
        rmse_dict["cr"][f"Pool_upsample_{n}_layers"] = {}

    
    for i in tqdm(range(max_reducation_range), unit = "reduction rank", desc = "Computing end point fitting", disable = not(verbose)):

        for n_layer in tqdm(range(n_layers),disable=not(verbose), unit = "layers", desc = "Computing AE layers"): 
            
            
            pooling_model = NoConvAE(n_layer, pooling_dim=pooling_dim, pooling_mode="Avg")

            min_max_idx = get_min_max_idx(test_ssp_arr, axs=1, pad=True)

            reduced_indices = get_interpolation_points_idx(min_max_idx, axis=1, reduction_rank=i)

            interpolated_points = np.take(test_ssp_arr, reduced_indices, axis=1)

            interpolated_points_tens = torch.tensor(interpolated_points)

            pooled_upsampled_interpolated_points = pooling_model(interpolated_points_tens).detach().numpy()

            cr = input_size/pooling_model.bottleneck.numel()

            interpolator = CubicSpline(reduced_indices, pooled_upsampled_interpolated_points, axis=1)

            interpolated_ssp_arr = interpolator(np.arange(test_ssp_arr.shape[1]))


            ssp_rmse = np.sqrt(np.mean((test_ssp_arr - interpolated_ssp_arr) ** 2))

            ecs_interpolated_idx = np.argmax(interpolated_ssp_arr,axis=1)
            ecs_interpolated = depth_array[ecs_interpolated_idx]

            ecs_rmse = np.sqrt(np.mean((ecs_truth - ecs_interpolated) ** 2))        

            min_max_idx_interpolated = get_min_max_idx(interpolated_ssp_arr, axis =1, pad=False)
            mean_number_error_min_max = np.mean(np.abs(np.sum(min_max_idx_truth,axis=1) - np.sum(min_max_idx_interpolated,axis=1)))

            F1_score = get_f1_score(min_max_idx_truth, min_max_idx_interpolated, axs=1, kernel_size=10)


            rmse_dict["SSP"][f"Pool_upsample_{n_layer}_layers"][f"reduction_rank_{i}"] = ssp_rmse
            rmse_dict["ECS"][f"Pool_upsample_{n_layer}_layers"][f"reduction_rank_{i}"] = ecs_rmse
            rmse_dict["mean_error_n_min_max"][f"Pool_upsample_{n_layer}_layers"][f"reduction_rank_{i}"] = mean_number_error_min_max
            rmse_dict["F1_score"][f"Pool_upsample_{n_layer}_layers"][f"reduction_rank_{i}"] = F1_score
            rmse_dict["cr"][f"Pool_upsample_{n_layer}_layers"][f"reduction_rank_{i}"] = cr



    with open(f'pickle/rmse_pca_all_components_with_pooling_upsampling_unorm_xp_{xp}.pkl', 'wb') as f:
        pickle.dump(rmse_dict, f)
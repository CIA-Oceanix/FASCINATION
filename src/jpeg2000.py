import sys
import os

running_path = "/Odyssey/private/o23gauvr/code/FASCINATION/"
sys.path.insert(0,running_path)
os.chdir(running_path)

import os 
import glymur
import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch
import hydra
from omegaconf import OmegaConf
from src.utils import loading_datamodule_phase, unorm_ssp_arr_3D
import tempfile


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

        if n > 0:
            self.decoder.append(nn.Upsample(size=None, mode=self.upsample_mode))
        else:
            self.decoder.append(nn.Identity())

    
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



def get_f1_score(min_max_idx_truth, min_max_idx_ae, axs=1, kernel_size=10):



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
    #raise a warning but not of importance

    # Compute f1_score and avoid division by zero when both precision and recall are 0
    sum_scores = precision_score + recall_score
    f1_score = np.where(sum_scores == 0, 0, 2 * (precision_score * recall_score) / sum_scores)

    return f1_score




if __name__ == "__main__":

    verbose = True
    xp="autoencoder_V2" #autoencoder_V2 #dense_ae
    pooling_dim = "spatial" if xp == "autoencoder_V2" else "all"

    max_ratio = 3500
    ratios = np.logspace(np.log10(0.1), np.log10(max_ratio), 100)  #np.linspace(1,max_ratio,3)  #np.linspace(1,3500,100) #np.logspace(-1, 6, num=100)
    n_layers = 5
    test_n_profiles = None #100 #None #100

    gpu=0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    cfg_path = f"config/xp/{xp}.yaml"
    cfg = OmegaConf.load(cfg_path)

    
    cfg.datamodule.n_profiles = None if test_n_profiles is None else 10*test_n_profiles


    print("Inititing datamodule; Generating train and test datasets")

    dm = hydra.utils.call(cfg.datamodule)

    test_ssp_arr, dm = loading_datamodule_phase(dm)

    test_ssp_arr_size = test_ssp_arr.nbytes
    input_size = test_ssp_arr.size
    input_shape = test_ssp_arr.shape
    time_size = input_shape[0]


    if  dm.norm_stats["norm_location"] == "datamodule":
        test_ssp_arr = unorm_ssp_arr_3D(test_ssp_arr, dm)


    depth_array = dm.depth_array

    ecs_truth_idx = np.argmax(test_ssp_arr,axis=1)
    ecs_truth = depth_array[ecs_truth_idx]

    min_max_idx_truth = get_min_max_idx(test_ssp_arr, pad=False)


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



    for n_layer in tqdm(range(n_layers),disable=not(verbose), unit = "layers", desc = "Computing AE layers"): 

        with torch.no_grad():
            pooling_model = NoConvAE(n_layer, pooling_dim=pooling_dim, pooling_mode="Avg")
            pooling_model.decoder[-1].size = input_shape[1:]

            test_ssp_tens = torch.tensor(test_ssp_arr).to(device)
            
            pooled_ssp_arr = pooling_model.encoder(test_ssp_tens.unsqueeze(1)).squeeze(1).detach().cpu().numpy()

            res = int(np.log(np.min(pooled_ssp_arr.shape))/np.log(2)+1)  #min(6,)

        for ratio in tqdm(ratios, desc="Ratios"):

            decompressed_ssp_arr = np.zeros(pooled_ssp_arr.shape)

            image_size = 0

            for img_idx in tqdm(range(time_size), desc="Images"):

                img = pooled_ssp_arr[img_idx]

                nan_min = np.nanmin(img)
                nan_max = np.nanmax(img)
                
                img = (255 * (img - nan_min) / (nan_max - nan_min)).astype(np.uint8)
                img = img.transpose(1, 2, 0)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    output_filename = os.path.join(tmpdirname, 'hyperspectral.jp2')
                    glymur.Jp2k(output_filename, data=img, cratios=[ratio], numres=res)
                    
                    #The number of resolutions (r) in JPEG2000 is tied to how many times the tile (or image) can be halved. 
                    # In each resolution level, the dimensions of a tile are effectively divided by 2. 
                    # To support r resolutions, the smallest dimension of a tile should be large enough so that after r – 1 successive halvings, 
                    # you still have at least 1 (preferably more) pixel. In practice this means that if your tile’s width (or height) is T, you typically need
                    # T ≥ 2^(r – 1)

                    # r does not need to be an integer, but it is typically an integer. 
                    # It does not have an impact on the compression ratio, but it does have an impact on the quality of the decompressed image.


                    image_size = image_size + os.stat(output_filename).st_size

                    # # Assuming the image is a numpy array with shape (height, width, channels)
                    # height, width = transposed_ssp_img.shape[0], transposed_ssp_img.shape[1]
                    # bpp = image_size * 8 / (width * height)

                    # Read back the compressed image.
                    rec = glymur.Jp2k(output_filename).read()

                uncompressed_img = (rec.astype(float) / 255) * (nan_max - nan_min) + nan_min
                uncompressed_img = uncompressed_img.transpose(2, 0, 1)

                decompressed_ssp_arr[img_idx] = uncompressed_img

            with torch.no_grad():
                decompressed_ssp_tens = torch.tensor(decompressed_ssp_arr).to(device)
                output_ssp_arr = pooling_model.decoder(decompressed_ssp_tens.unsqueeze(1)).squeeze(1).detach().cpu().numpy()

            cr = input_size / image_size

            ssp_rmse = np.sqrt(np.mean((test_ssp_arr - output_ssp_arr) ** 2))

            ecs_interpolated_idx = np.argmax(output_ssp_arr,axis=1)
            ecs_interpolated = depth_array[ecs_interpolated_idx]

            ecs_rmse = np.sqrt(np.mean((ecs_truth - ecs_interpolated) ** 2))        

            min_max_idx_interpolated = get_min_max_idx(output_ssp_arr, axs=1, pad=False)
            mean_number_error_min_max = np.mean(np.abs(np.sum(min_max_idx_truth,axis=1) - np.sum(min_max_idx_interpolated,axis=1)))

            F1_score = get_f1_score(min_max_idx_truth, min_max_idx_interpolated, axs=1, kernel_size=10)
            f1_score = np.mean(F1_score)        

            rmse_dict["SSP"][f"Pool_upsample_{n_layer}_layers"][f"compression ratio {ratio}"] = ssp_rmse
            rmse_dict["ECS"][f"Pool_upsample_{n_layer}_layers"][f"compression ratio {ratio}"] = ecs_rmse
            rmse_dict["mean_error_n_min_max"][f"Pool_upsample_{n_layer}_layers"][f"compression ratio {ratio}"] = mean_number_error_min_max
            rmse_dict["F1_score"][f"Pool_upsample_{n_layer}_layers"][f"compression ratio {ratio}"] = f1_score
            rmse_dict["cr"][f"Pool_upsample_{n_layer}_layers"][f"compression ratio {ratio}"] = cr



        
    with open(f'pickle/rmse_jpeg_2000.pkl', 'wb') as f:
        pickle.dump(rmse_dict, f)

    
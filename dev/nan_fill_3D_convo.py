import numpy as np
import torch 
import torch.nn.functional as F
from typing import Union, List
from tqdm import tqdm
import xarray as xr

def nan_fill_3D_weighted_mean(tensor: torch.tensor, 
                            weight: Union[None,torch.tensor] = None,
                            kernel_size=3):
    

        
        
    # if weight.dtype != torch.float32 and weight.dtype != torch.float64:
    #     raise TypeError("Input tensor must have float dtype")
    
    
    nan_indices = torch.nonzero(torch.isnan(tensor.to('cpu')), as_tuple=False)
    
    for idx in tqdm(nan_indices, desc = "Weighted mean interpolation"):
        kernel = kernel_size
        tensor[idx] = weighted_mean(tensor,weight, idx, kernel)
        
    return tensor


def weighted_mean(tensor,weight, idx, kernel_size):
    
        if weight is None:
            weight = torch.ones(kernel_size, kernel_size, kernel_size).double().to(tensor.device)
        
        if kernel_size > 3:
            print(idx,kernel_size)
        
        ipt = expand_tensor_padding(tensor[idx[0],:,:,:],kernel_size)

        
        block =  ipt[idx[1]:idx[1]+kernel_size,
                     idx[2]:idx[2]+kernel_size,
                     idx[3]:idx[3]+kernel_size]


        block = block.nan_to_num()
        weight[block == 0] = 0

        conv_result = F.conv3d(block.unsqueeze(0).unsqueeze(0),
                               weight.unsqueeze(0).unsqueeze(0),
                               padding="valid").squeeze()
        
        
        
        conv_result = conv_result/weight.sum()
        
        if conv_result.isnan():
            weighted_mean(tensor,weight, idx, kernel_size + 1)
            
        else:
            return conv_result
        
        
        
def expand_tensor_padding(tensor, kernel_size):
    # Get the shape of the original tensor
    original_shape = tensor.shape
    
    # Compute the amount of padding to add at the beginning and end of each dimension
    padding = kernel_size // 2
    
    # Define the new shape with padding added along the last three dimensions
    new_shape = np.array(original_shape)+ np.array([kernel_size,kernel_size,kernel_size])
    
    # Create a new tensor with zeros of the new shape
    expanded_tensor = torch.zeros(list(new_shape), dtype=tensor.dtype, device=tensor.device)
    
    # Calculate the slice indices for copying the original tensor into the expanded tensor
    slices = tuple(slice(padding, padding + original_shape[i]) for i in range(0, 3))
    
    # Copy the original tensor into the expanded tensor
    expanded_tensor[slices] = tensor
    
    return expanded_tensor


if __name__ == '__main__':
    
    # # Randomly choose sizes for each dimension between 5 and 10
    # sizes = [np.random.randint(5, 10) for _ in range(4)]

    # # Create the tensor with random dimensions
    # tensor = torch.randn(*sizes).double()

    # # Choose arbitrary indices for NaN values
    # nan_indices = [(np.random.randint(0, sizes[0]-1), np.random.randint(0, sizes[1]-1), 
    #                 np.random.randint(0, sizes[2]-1), np.random.randint(0, sizes[3]-1)) for _ in range(3)]

    # # Assign NaN values
    # for index in nan_indices:
    #     tensor[index] = float('nan')
  
    # kernel_size = 3      
    # weight = torch.ones(kernel_size, kernel_size, kernel_size).double().to(tensor.device)
    
    # ##* .detach().clone() because torch tensor are mutable objects
    # tensor_filled = nan_fill_3D_weighted_mean(tensor.detach().clone())
    gpu = 3
    if torch.cuda.is_available() and gpu is not None:
    ##This may not be necessary outside the notebook
        dev = f"cuda:{gpu}"
    else:
        dev = "cpu"
    
    device = torch.device(dev)
    ###! issue with INT_MAX argument when working on gpu


    print("Selected device:", device)
    
    sound_speed_path = "/DATASET/eNATL/eNATL60_BLB002_sound_speed_regrid_0_1000m.nc"
    input_da = xr.open_dataarray(sound_speed_path)
    coords = input_da.coords
    tensor = torch.tensor(input_da.values).to(device)
    del input_da
    
    nan_number = torch.isnan(tensor).sum().item()
    print(nan_number, "nan")
    print(np.round(nan_number*100/2249568000,2), "%")
    
    nan_fill_3D_weighted_mean(tensor)
    ##* It should be ok to use the mutuable aspect of tensor to avoid cloning and save memory
    
    
    nan_number = torch.isnan(tensor).sum().item()
    print(nan_number, "nan")
    print(np.round(nan_number*100/2249568000,2), "%")
    
    if torch.isnan(tensor).any():
        #raise ValueError("All nan values haven't been filled")
        print("All nan values haven't been filled")

    sound_speed_path_nan_filled = '/DATASET/envs/o23gauvr/ss_depth_features_weighted_mean_filled.nc' 
    
    da_to_save = xr.DataArray(data=tensor.to('cpu').detach().numpy(), coords=coords)   
    del tensor
    da_to_save.to_netcdf(sound_speed_path_nan_filled)
    
    del da_to_save
    da = xr.open_dataarray(sound_speed_path_nan_filled)
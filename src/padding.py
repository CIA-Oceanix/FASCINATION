import torch



def linear_padding(self, x, interp_size = 20):

    first_2_points = x[:,:2,:,:]  
    last_2_points = x[:,-2:,:,:]  
    
    start_pad = torch.zeros(x.size(0), interp_size, x.size(2), x.size(3), device = x.device)  
    end_pad = torch.zeros(x.size(0), interp_size, x.size(2), x.size(3), device = x.device)  
    
    
    for i in range(0,interp_size):
        start_pad[:, -(i+1),:,:] = first_2_points[:, 0,:,:] + (i+1) * (first_2_points[:, 0,:,:] - first_2_points[:, 1,:,:])
        end_pad[:, i,:,:] = last_2_points[:, -1,:,:] + (i+1) * (last_2_points[:, -1,:,:] - last_2_points[:, -2,:,:]) 

    padded_x = torch.cat([start_pad, x, end_pad], dim=1) 
    
    return padded_x



def cubic_padding_multi_dim(self, signal, interp_size, dim=1):
    # interp_size: Number of points to pad at the beginning and end along the specified dimension
    # dim: The dimension along which to perform the padding (default is 1)
    
    # Move the dimension of interest to the first dimension
    signal = signal.transpose(dim, 0)  # Now signal has shape [107, 255, 174, 240]
    original_shape = signal.shape
    L = original_shape[0]  # Length of the dimension along which to pad (107)
    other_dims = original_shape[1:]  # (255, 174, 240)
    #N = torch.prod(torch.tensor(other_dims)).item()  # Total number of signals (255*174*240)
    N = other_dims.numel()
    
    # Flatten the rest of the dimensions
    signal = signal.reshape(L, N)  # Now signal is of shape [107, N]
    
    # Indices along the dimension
    x = torch.arange(L, device=signal.device, dtype=signal.dtype)  # Shape [107]
    
    # # Edge indices
    #edge_indices = torch.arange(interp_size, device=signal.device, dtype=torch.long)
    
    # Beginning of the signal
    x_start = x[:interp_size]  # Shape [interp_size]
    #y_start = signal[edge_indices, :]  # Shape [interp_size, N]
    y_start = signal[:interp_size,:]
    
    
    # Fit cubic polynomial at the start
    X_start = torch.stack([x_start**3, x_start**2, x_start, torch.ones_like(x_start)], dim=1)  # Shape [interp_size, 4]
    coeffs_start = torch.linalg.lstsq(X_start, y_start).solution  # Shape [4, N]
    
    # Generate padding for the start
    x_pad_start = x_start - interp_size  # Extrapolate to positions before the signal
    X_pad_start = torch.stack([x_pad_start**3, x_pad_start**2, x_pad_start, torch.ones_like(x_pad_start)], dim=1)  # Shape [interp_size, 4]
    y_pad_start = X_pad_start @ coeffs_start  # Shape [interp_size, N]
    
    # End of the signal
    x_end = x[-interp_size:]  # Shape [interp_size]
    y_end = signal[-interp_size:, :]  # Shape [interp_size, N]
    
    # Fit cubic polynomial at the end
    X_end = torch.stack([x_end**3, x_end**2, x_end, torch.ones_like(x_end)], dim=1)  # Shape [interp_size, 4]
    coeffs_end = torch.linalg.lstsq(X_end, y_end).solution  # Shape [4, N]
    
    # Generate padding for the end
    x_pad_end = x_end + interp_size  # Extrapolate to positions after the signal
    X_pad_end = torch.stack([x_pad_end**3, x_pad_end**2, x_pad_end, torch.ones_like(x_pad_end)], dim=1)  # Shape [interp_size, 4]
    y_pad_end = X_pad_end @ coeffs_end  # Shape [interp_size, N]
    
    # Concatenate padding and original signal
    padded_signal = torch.cat([y_pad_start, signal, y_pad_end], dim=0)  # Shape [L + 2*interp_size, N]
    
    # Reshape back to original dimensions, adjusting for the new size of the padded dimension
    padded_L = L + 2 * interp_size
    padded_signal = padded_signal.reshape(padded_L, *other_dims)  # Shape [padded_L, 255, 174, 240]
    
    # Move the dimension back to its original position
    dims = list(range(padded_signal.dim()))
    dims[0], dims[dim] = dims[dim], dims[0]
    padded_signal = padded_signal.permute(*dims)
    
    return padded_signal
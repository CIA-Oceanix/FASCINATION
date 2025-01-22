
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class AE_CNN_3D_Encoder(nn.Module):
    
    def __init__(self,
                 channels_list: list,
                 kernel_list: list,
                 n_conv_per_layer: int,
                 padding: Union[int, str], 
                 act_fn: object,
                 pooling_layer: torch.nn,
                 pooling_dim: str,
                 linear_layer: bool,
                 latent_size: int,
                 dropout_proba: float,
                 dtype: object):

    
        super().__init__()
    
                
        num_layers = len(channels_list)-1
        
        self.linear_layer = linear_layer
        self.latent_size = latent_size
        self.pooling_dim = pooling_dim


        
        # if isinstance(pooling_layer, nn.Identity):
        #     kernel_list = [7]*num_layers
        
        # else: 
        #     kernel_list = [7,7,5,5,3,3]
        #     kernel_list = kernel_list + [3]*(num_layers-len(kernel_list))
            
            
        layers = []    
            
        for i in range(num_layers):
            
            ker = kernel_list[i]
            
            if not isinstance(padding, int):
                pad = (ker - 1)//2


            if pooling_dim == "all":
                pooling_layer.kernel_size = ker
                pooling_layer.padding = pad
                
            elif pooling_dim == "depth":
                ker = (ker,1,1)
                pad = (pad,0,0)
                pooling_layer.kernel_size = ker
                pooling_layer.padding = pad

            elif pooling_dim == "spatial":
                ker = (1,ker,ker)
                pad = (0, pad, pad)
                pooling_layer.kernel_size = ker
                pooling_layer.padding = pad
    
             
            for j in range(n_conv_per_layer):
                
                if j == 0:
                    in_ch = channels_list[i]
                else:
                    in_ch = channels_list[i+1]
            
                conv_layer = nn.Conv3d(in_channels = in_ch, out_channels = channels_list[i+1], kernel_size = ker, stride = 1, padding = pad, dtype = dtype)
                layers.append(conv_layer)
        
            layers.extend(
                [pooling_layer, act_fn, nn.Dropout(p=dropout_proba, inplace=False)]
            )
        
            
        self.net = nn.Sequential(*layers)
        
        # nn.init.constant_(self.net[0].bias, 0)
        # self.net[0].weight = torch.nn.parameter.Parameter(torch.tensor([[[0.,0.,0.,1.,0.,0.,0.]]]))

                
        
    def forward(self, x):
        

        z = self.net(x)
        
        if self.linear_layer:   
            if self.pooling_dim == "depth":
                depth_size = z.shape[2]
                n_channels = z.shape[1]
                z = z.permute(0, 3, 4, 1, 2) 
                z = z.reshape(-1,n_channels*depth_size)
                z = nn.Linear(n_channels*depth_size, self.latent_size, device=z.device)(z)

            
            if self.pooling_dim == "spatial":
                spatial_size = z.shape[-2:]
                n_channels = z.shape[1]
                z = z.transpose(1 , 2) 
                z = z.reshape(-1,n_channels*spatial_size.numel())
                z = nn.Linear(n_channels*spatial_size.numel(), self.latent_size, device=z.device)(z)
                

            
            elif self.pooling_dim == "all":
     
                z = nn.Flatten()(z)
                z = nn.Linear(z.shape[-1], self.latent_size, device=z.device)(z)

       
        return z

    
    
    
    
class AE_CNN_3D_Decoder(nn.Module):
    
    def __init__(self,
                 channels_list: list,
                 kernel_list: list,
                 n_conv_per_layer: int,
                 padding: Union[int, str], 
                 pooling_dim: str,
                 linear_layer: bool,
                 latent_size: int,
                 act_fn: object,
                 final_act_fn: object,
                 final_upsample_str: str,
                 final_upsample_mode: str,
                 upsample_layer: torch.nn,
                 dtype: object):
    
    
        super().__init__()
    

        
        num_layers = len(channels_list)-1
        
        self.channels_list = channels_list
        self.final_upsample_str = final_upsample_str
        self.upsample_layer = upsample_layer
        self.linear_layer = linear_layer
        self.latent_size = latent_size
        self.final_upsample_mode = final_upsample_mode
        self.pooling_dim = pooling_dim
        
        layers = []
        
        kernel_list = kernel_list[::-1]
        channels_list = channels_list[::-1]
  
        

        for i in range(num_layers):
            
            ker = kernel_list[i]

            if not isinstance(padding, int):
                pad = (ker - 1)//2


            if pooling_dim == "all":
                upsample_layer.kernel_size = ker
                upsample_layer.padding = pad
                
            elif pooling_dim == "depth":
                ker = (ker,1,1)
                pad = (pad,0,0)
                upsample_layer.kernel_size = ker
                upsample_layer.padding = pad

            elif pooling_dim == "spatial":
                ker = (1,ker,ker)
                pad = (0, pad, pad)
                upsample_layer.kernel_size = ker
                upsample_layer.padding = pad

            
            for j in range(n_conv_per_layer):
                
                if j == 0:
                    in_ch = channels_list[i]
                
                else:
                    in_ch = channels_list[i+1]
                    
            

                conv_transpose_layer = nn.ConvTranspose3d(in_channels = in_ch, out_channels = channels_list[i+1],  kernel_size=ker, stride=1, padding=pad, dtype = dtype)

                layers.append(conv_transpose_layer)
                
            layers.extend([upsample_layer, act_fn])
                
            if i == (num_layers-1):
                layers[-1] = final_act_fn
            
            
        # ker = kernel_list[-num_layers+1]
        
        # if not isinstance(padding, int):
        #     pad = (ker - 1)//2
            
        # layers.extend([nn.ConvTranspose3d(in_channels = channels_list[1], out_channels = channels_list[0],  kernel_size=ker, stride=1, padding=pad, dtype = dtype)]*n_conv_per_layer +
        #               [final_upsample_layer, final_act_fn])
        
        
        self.net = nn.Sequential(*layers)

        
        # nn.init.constant_(self.net[0].bias, 0)
        # self.net[0].weight = torch.nn.parameter.Parameter(torch.tensor([[[0.,0.,0.,1.,0.,0.,0.]]]))  
        
              
    def forward(self, z, x_hat_spatial_shape, z_spatial_shape = []):
        
        if not (isinstance(self.upsample_layer, nn.Identity)):  #? necessary condition ?
        
            if self.final_upsample_str == "upsample_pooling":
                self.net[-2] = nn.Sequential(self.upsample_layer,
                                             nn.AdaptiveMaxPool3d(output_size = x_hat_spatial_shape))
                
            elif self.final_upsample_str == "upsample":
                self.net[-2] = nn.Upsample(size = x_hat_spatial_shape, mode = self.final_upsample_mode)
                
            
        if self.linear_layer:

            if self.pooling_dim == "depth": 

                z = nn.Linear(self.latent_size, z_spatial_shape[1]*z_spatial_shape[2], device=z.device)(z)

                z = z.reshape(z_spatial_shape[0],z_spatial_shape[3],z_spatial_shape[4],z_spatial_shape[1],z_spatial_shape[2])

                z = z.permute(0,3,4,1,2)

            if self.pooling_dim == "spatial":

                z =  nn.Linear(self.latent_size, z_spatial_shape[1]*z_spatial_shape[3]*z_spatial_shape[4], device=z.device)(z)

                z = z.reshape(z_spatial_shape[0],z_spatial_shape[2],z_spatial_shape[1],z_spatial_shape[3],z_spatial_shape[4])

                z = z.transpose(1,2)


            elif self.pooling_dim == "all":
                z = nn.Linear(self.latent_size, z_spatial_shape[1:].numel(), device=z.device)(z)

                z = nn.Unflatten(dim=1, unflattened_size=(z_spatial_shape[1:]))(z)

        
        return self.net(z)
    

        


class AE_CNN_3D(nn.Module):
    
    def __init__(self,
                 channels_list: list = [1,1,1,1],
                 n_conv_per_layer: int = 1,
                 padding: Union[int, str] = "same",
                 interp_size: int = 20,
                 act_fn_str : str = "Relu",
                 final_act_fn_str: str = "linear",
                 final_upsample_str: str = "upsample",
                 upsample_mode: str = "trilinear",
                 pooling: bool = "Max",
                 pooling_dim: str = "all",
                 linear_layer: bool = True,
                 latent_size: int = 9,
                 dropout_proba: bool = 0,
                 dtype_str: str = "float32"):
        

    
        super().__init__()


        num_layers = len(channels_list)-1


        kernel_list = [7,7,5,5,3,3]
        #kernel_list = [3,3,3,3,3,3]
        kernel_list = kernel_list + [3]*(num_layers-len(kernel_list))
        kernel_list = kernel_list[:num_layers]


        if pooling_dim == "all":
            pool_str = 2
            
        elif pooling_dim == "depth":
            pool_str = (2,1,1)

        elif pooling_dim == "spatial":
            pool_str = (1,2,2)

        elif pooling_dim == "None":
            pool_str = 1
            channels_list = [1]*len(channels_list)


            

        pooling_dict = {"Avg": nn.AvgPool3d(kernel_size= 1,stride=pool_str, padding = 0),
                        "Max": nn.MaxPool3d(kernel_size=1, stride=pool_str, padding = 0),
                        "None": nn.Identity()}     
        
        upsample_dict = {"Avg": nn.Upsample(scale_factor = pool_str, mode = upsample_mode),
                        "Max":  nn.Upsample(scale_factor = pool_str, mode = upsample_mode),
                        "None": nn.Identity()}       
        
        act_fn_dict = {"Sigmoid": nn.Sigmoid(),
                       "Tanh": nn.Tanh(),
                       "Softmax": nn.Softmax(dim=1),
                       "Relu": nn.ReLU(),
                       "Elu": nn.ELU(),
                       "Gelu": nn.GELU(),
                       "None": nn.Identity()}
        #"Linear": nn.Linear(0,0) impossible d'allouer la mémoire sufisante pour un e fonction lineaire (n_feature*depth*height*width)**2
        
        
        self.model_dtype = getattr(torch, dtype_str)
        self.linear_layer = linear_layer
        
        self.first_channel = channels_list[0]



        if interp_size == 0: # or pooling_dim == "spatial" or pooling_dim == "None":
            self.padding = None
        
        
        elif padding in ["linear", "cubic", "reflect", "replicate", "circular"]:
            self.padding = padding
            self.interp_size = interp_size
            
        else: 
            self.padding = None
        
        
        pooling_layer = pooling_dict[pooling]
        upsample_layer = upsample_dict[pooling]
        act_fn = act_fn_dict[act_fn_str]
        final_act_fn = act_fn_dict[final_act_fn_str]
        
        

        self.encoder = AE_CNN_3D_Encoder(channels_list = channels_list,
                                         kernel_list = kernel_list,
                                         n_conv_per_layer = n_conv_per_layer,
                                         padding = padding,
                                         act_fn = act_fn,
                                         pooling_layer = pooling_layer,
                                         pooling_dim = pooling_dim,
                                         linear_layer = linear_layer,
                                         latent_size = latent_size,
                                         dropout_proba =dropout_proba,
                                         dtype = self.model_dtype
                                         )        
        
        
        self.decoder = AE_CNN_3D_Decoder(channels_list = channels_list,
                                         kernel_list = kernel_list,
                                         n_conv_per_layer = n_conv_per_layer,
                                         padding = padding,
                                         pooling_dim = pooling_dim,
                                         linear_layer = linear_layer,
                                         latent_size = latent_size,
                                         act_fn = act_fn,
                                         final_act_fn = final_act_fn,
                                         final_upsample_str = final_upsample_str,
                                         final_upsample_mode=upsample_mode,
                                         upsample_layer = upsample_layer,
                                         dtype = self.model_dtype
                                         )        
    
    
    def forward(self, x):
        
        
        if self.padding == "linear":
            x = self.linear_padding(x, interp_size=self.interp_size)
            
        elif self.padding == "cubic":
            x = self.linear_padding(x, interp_size=self.interp_size)
            
        elif self.padding in ["reflect", "replicate", "circular"]:
            x = torch.nn.functional.pad(x, pad=(0, 0, 0, 0, self.interp_size, self.interp_size), mode=self.padding)
            
        ##TODO: see effect of padding only on top of signal
                    
            
        x_hat_spatial_shape = x.shape[1:]

        x = x.unsqueeze(1)  #.repeat(1,self.first_channel,1,1,1)
        
        z = self.encoder(x)
        
        self.bottleneck_shape = z.shape

        if self.linear_layer:
            z_pre_linear_shape = self.encoder.net(x).shape
        else:
            z_pre_linear_shape = []
        
        x_hat = self.decoder(z, x_hat_spatial_shape, z_pre_linear_shape)
              
        x_hat = x_hat.squeeze(1)  #x_hat[:,0,:,:,:].squeeze(1)
        
        if self.padding is not None:
            x_hat = x_hat[:,self.interp_size:-self.interp_size]

        
        return x_hat
    
    
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
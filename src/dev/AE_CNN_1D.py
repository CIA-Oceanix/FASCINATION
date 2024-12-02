
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class AE_CNN_1D_Encoder(nn.Module):
    
    def __init__(self,
                 channels_list: list,
                 n_conv_per_layer: int,
                 padding: int | str, 
                 act_fn: object,
                 pooling_layer: torch.nn,
                 linear_layer: bool,
                 latent_size: int,
                 dropout_proba: float,
                 dtype: object):

    
        super().__init__()
    
                
        num_layers = len(channels_list)

        
        if padding == "linear":
            depth = 107 + 2*5
        else:
            depth = 107
        
        if isinstance(pooling_layer, nn.Identity):
            kernel_list = [7]*num_layers
        
        else: 
            kernel_list = [7,7,5,5,3,3]
            kernel_list = kernel_list + [3]*(num_layers-len(kernel_list))

        
        layers = []    
            
        for i in range(num_layers-1):
            
            ker = kernel_list[i]
            
            if not isinstance(padding, int):
                pad = (ker - 1)//2

             
            for j in range(n_conv_per_layer):
                
                if j == 0:
                    in_ch = channels_list[i]
                else:
                    in_ch = channels_list[i+1]
            
                conv_layer = nn.Conv1d(in_channels = in_ch, out_channels = channels_list[i+1], kernel_size = ker, stride = 1, padding = pad, dtype = dtype)
                layers.append(conv_layer)
        
            layers.extend(
                [pooling_layer, act_fn, nn.Dropout(p=dropout_proba, inplace=False)]
            )
        
        
        if linear_layer and not(isinstance(pooling_layer, nn.Identity)):
            lenght = 107
            for i in range(num_layers - 1):
                lenght = 1+(lenght-1)//2
                
            layers.append([nn.Flatten(),
                           nn.Linear(channels_list[-1] * lenght, latent_size)])
        

            
        self.net = nn.Sequential(*layers)
        
        # nn.init.constant_(self.net[0].bias, 0)
        # self.net[0].weight = torch.nn.parameter.Parameter(torch.tensor([[[0.,0.,0.,1.,0.,0.,0.]]]))

        
                
        
    def forward(self, x):

        return self.net(x)

    
    
    
    
class AE_CNN_1D_Decoder(nn.Module):
    
    def __init__(self,
                 channels_list: list,
                 n_conv_per_layer: int,
                 padding: int | str, 
                 linear_layer: bool,
                 latent_size: int,
                 act_fn: object,
                 final_act_fn: object,
                 final_upsample_str: str,
                 pooling_layer: torch.nn,
                 dtype: object):
    
    
        super().__init__()
    

        
        num_layers = len(channels_list)
                
        if padding == "linear":
            depth = 107 + 2*5
        else:
            depth = 107

        
        if isinstance(pooling_layer, nn.Identity):
            upsample_layer = nn.Identity()
            final_upsample_layer = nn.Identity()
            
            kernel_list = [7]*num_layers
                        

            
        else:
            upsample_layer = nn.Upsample(scale_factor = 2)
            
            if final_upsample_str == "upsample_pooling":
                final_upsample_layer = [nn.Upsample(scale_factor = 2),
                                        nn.AdaptiveMaxPool1d(output_size = depth)]
                
            elif final_upsample_str == "upsample":
                final_upsample_layer = [nn.Upsample(size=depth)]
            


            kernel_list = [7,7,5,5,3,3]
            kernel_list = kernel_list + [3]*(num_layers-len(kernel_list))


            
            
        
        
        if linear_layer and not(isinstance(pooling_layer, nn.Identity)):
            lenght = depth
            for i in range(num_layers - 1):
                lenght = 1+(lenght-1)//2
                
            layers = [
                [nn.Linear(latent_size, channels_list[0]*lenght),
                 nn.Unflatten(dim=1, unflattened_size=(int(channels_list[0]), lenght))]
                ] 

        
        else:
            layers = []
        

        for i in range(1, num_layers - 1):
            
            ker = kernel_list[-i]
            
            if not isinstance(padding, int):
                pad = (ker - 1)//2
            
            for j in range(n_conv_per_layer):
                
                if j == 0:
                    in_ch = channels_list[-i]
                
                else:
                    in_ch = channels_list[-i-1]
                    
            

                conv_transpose_layer = nn.ConvTranspose1d(in_channels = in_ch, out_channels = channels_list[-i-1],  kernel_size=ker, stride=1, padding=pad, dtype = dtype)

                layers.append(conv_transpose_layer)
        
            layers.extend([upsample_layer, act_fn])
            
        ker = kernel_list[-num_layers+1]
        
        if not isinstance(padding, int):
            pad = (ker - 1)//2
            
        layers.extend([nn.ConvTranspose1d(in_channels = channels_list[1], out_channels = channels_list[0],  kernel_size=ker, stride=1, padding=pad, dtype = dtype)]*n_conv_per_layer +
                      final_upsample_layer +
                      [final_act_fn])
        
        
        self.net = nn.Sequential(*layers)

        
        # nn.init.constant_(self.net[0].bias, 0)
        # self.net[0].weight = torch.nn.parameter.Parameter(torch.tensor([[[0.,0.,0.,1.,0.,0.,0.]]]))  
        
              
    def forward(self, x):
        return self.net(x)
    

        


class AE_CNN_1D(nn.Module):
    
    def __init__(self,
                 channels_list: list = [1,1,1,1],
                 n_conv_per_layer: int = 1,
                 padding: int | str = "same",
                 act_fn_str : str = "Relu",
                 final_act_fn_str: str = "Linear",
                 final_upsample_str: str = "upsample",
                 pooling: bool = "Max",
                 linear_layer: bool = True,
                 dropout_proba: bool = 0,
                 dtype_str: str = "float32"):
        

    
        super().__init__()
        


        pooling_dict = {"Avg": nn.AvgPool1d(kernel_size= 3,stride=2, padding = 1),
                        "Max": nn.MaxPool1d(kernel_size=3, stride=2, padding = 1),
                        "None": nn.Identity()}       
        
        act_fn_dict = {"Sigmoid": nn.Sigmoid(),
                       "Tanh": nn.Tanh(),
                       "Softmax": nn.Softmax(dim=1),
                       "Relu": nn.ReLU(),
                       "Gelu": nn.GELU(),
                       "Linear": nn.Linear(107,107),
                       "None": nn.Identity()}
        
        
        self.model_dtype = getattr(torch, dtype_str)
        
        self.padding = padding
        
        
        pooling_layer = pooling_dict[pooling]
        act_fn = act_fn_dict[act_fn_str]
        final_act_fn = act_fn_dict[final_act_fn_str]
        
        latent_size = None
        if linear_layer:
            latent_size = 9
        

        self.encoder = AE_CNN_1D_Encoder(channels_list = channels_list,
                                         n_conv_per_layer = n_conv_per_layer,
                                         padding = padding,
                                         act_fn = act_fn,
                                         pooling_layer = pooling_layer,
                                         linear_layer = linear_layer,
                                         latent_size = latent_size,
                                         dropout_proba =dropout_proba,
                                         dtype = self.model_dtype
                                         )        
        
        
        self.decoder = AE_CNN_1D_Decoder(channels_list = channels_list,
                                         n_conv_per_layer = n_conv_per_layer,
                                         padding = padding,
                                         linear_layer = linear_layer,
                                         latent_size = latent_size,
                                         act_fn = act_fn,
                                         final_act_fn = final_act_fn,
                                         final_upsample_str = final_upsample_str,
                                         pooling_layer = pooling_layer,
                                         dtype = self.model_dtype
                                         )        
    
    
    def forward(self, x):
        
        if self.padding == "linear":
            x = self.linear_padding(x)

        x = x.unsqueeze(1)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
              
        x_hat = x_hat.squeeze(1)
        
        if self.padding == "linear":
            x_hat = x_hat[:,5:-5]

        
        return x_hat
    
    
    def linear_padding(self, x, interp_size = 5):

        first_2_points = x[:, :2]  
        last_2_points = x[:, -2:]  
        
        start_pad = torch.zeros(x.size(0), interp_size, device = x.device)  
        end_pad = torch.zeros(x.size(0), interp_size, device = x.device)  
        
        
        for i in range(0,interp_size):
            start_pad[:, -(i+1)] = first_2_points[:, 0] + (i+1) * (first_2_points[:, 0] - first_2_points[:, 1])
            end_pad[:, i] = last_2_points[:, -1] + (i+1) * (last_2_points[:, -1] - last_2_points[:, -2]) 

        padded_x = torch.cat([start_pad, x, end_pad], dim=1) 
        
        return padded_x
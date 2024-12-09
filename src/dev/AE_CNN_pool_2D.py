
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class AE_CNN_2D_pool_Encoder(nn.Module):
    
    def __init__(self,
                 input_channels: int,
                 latent_dim: int,
                 num_layers: int,
                 act_fn: object,
                 pooling_layer: str,
                 dropout_proba: float,
                 init_params: tuple,
                 dtype: object):
    
    
        super().__init__()
    
        
        channels_list = np.linspace(input_channels, latent_dim, num = num_layers + 1, dtype = np.int16)
        layers = []
        
            
            
        for i in range(num_layers):
        
            layers.append(
                [
                    nn.Conv2d(in_channels = channels_list[i], out_channels = channels_list[i+1], kernel_size = (1,1), dtype = dtype),
                    pooling_layer,
                    act_fn,
                    nn.Dropout(p=dropout_proba, inplace=False),                   
                ]
            )
            
        
        self.net = nn.Sequential(*sum(layers, []))
        
        if init_params["use"]: 
            init_weight = init_params["params"]["weight"]
            nn.init.constant_(self.net[0].bias, 0)
            #self.net[0].bias = torch.nn.parameter.Parameter(-torch.tensor(init_bias, dtype=self.net[0].weight.dtype, device = self.net[0].weight.device))
            self.net[0].weight = torch.nn.parameter.Parameter(torch.tensor(init_weight[:channels_list[1],:], dtype=self.net[0].weight.dtype, device = self.net[0].weight.device).unsqueeze(-1).unsqueeze(-1))
        
        
    def forward(self, x):
        
            
        return self.net(x)
    
    
    
    
    
class AE_CNN_2D_pool_Decoder(nn.Module):
    
    def __init__(self,
                 input_channels: int,
                 latent_dim: int,
                 num_layers: int,
                 act_fn: object,
                 final_act_fn: object,
                 pooling_layer: object,
                 init_params: tuple,
                 dtype: object):
    
    
        super().__init__()
    
        
        channels_list = np.linspace(latent_dim, input_channels, num = num_layers + 1, dtype = np.int16)
        layers = []
        
        
        if isinstance(pooling_layer, nn.Identity):
            upsample_layer = nn.Identity()
            
        else:
            upsample_layer = nn.Upsample(scale_factor = (2,2))
            
        
    
        for i in range(num_layers):
        
            layers.append(
                [
                    nn.ConvTranspose2d(in_channels = channels_list[i], out_channels = channels_list[i+1], kernel_size = (1,1), dtype = dtype),
                    upsample_layer,
                    act_fn                   
                ]
            )
            
        
        layers[-1][-1] == final_act_fn
        self.net = nn.Sequential(*sum(layers, []))
        
        if init_params["use"]: 
            init_weight = init_params["params"]["weight"]
            nn.init.constant_(self.net[-3].bias, 0)  ##? 0 or -1 ?
            #self.net[0].bias = torch.nn.parameter.Parameter(-torch.tensor(init_bias, dtype=self.net[0].weight.dtype, device = self.net[0].weight.device))
            self.net[-3].weight = torch.nn.parameter.Parameter(torch.tensor(init_weight[:channels_list[-2],:], dtype=self.net[-3].weight.dtype, device = self.net[-3].weight.device).unsqueeze(-1).unsqueeze(-1))
        
        
        
        
    def forward(self, x):
        return self.net(x)
    
    
        


class AE_CNN_pool_2D(nn.Module):
    
    def __init__(self,
                 input_channels: int = 107,
                 latent_dim: int = 4,
                 num_layers: int = 4,
                 act_fn_str : str = "Relu",
                 final_act_fn_str: str = "Sigmoid",
                 pooling_str: str = "None",
                 dropout_proba: float = 0.1,
                 init_params: tuple = (None, None),
                 dtype_str: str = "float32"):

    
        super().__init__()
        

        
        pooling_dict = {"Avg": nn.AvgPool2d(kernel_size=2),
                        "Max": nn.MaxPool2d(kernel_size=2),
                        "None": nn.Identity()}
        
        
        act_fn_dict = {"Sigmoid": nn.Sigmoid(),
                       "Tanh": nn.Tanh(),
                       "Softmax": nn.Softmax(dim=1),
                       "Relu": nn.ReLU(),
                       "Gelu": nn.GELU(),
                       "None": nn.Identity()}
        
        
        self.model_dtype = getattr(torch, dtype_str)
        self.init_params = init_params
        
        pooling_layer = pooling_dict[pooling_str]
        act_fn = act_fn_dict[act_fn_str]
        final_act_fn = act_fn_dict[final_act_fn_str]
        


        self.encoder = AE_CNN_2D_pool_Encoder(input_channels = input_channels,
                                              latent_dim = latent_dim,
                                              num_layers = num_layers,
                                              act_fn = act_fn,
                                              pooling_layer = pooling_layer,
                                              dropout_proba = dropout_proba,
                                              init_params = init_params,
                                              dtype = self.model_dtype
                                              )        
        
        
        self.decoder = AE_CNN_2D_pool_Decoder(input_channels = input_channels,
                                              latent_dim =latent_dim,
                                              num_layers = num_layers,
                                              act_fn = act_fn,
                                              final_act_fn = final_act_fn,
                                              pooling_layer = pooling_layer,
                                              init_params = init_params,
                                              dtype = self.model_dtype
                                              )
        
    
    
    def forward(self, x):
        
        if self.init_params["use"]:
            x = x - torch.tensor(self.init_params["params"]["bias"].reshape(1,-1,1,1), dtype = x.dtype, device = x.device)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        if self.init_params["use"]:
            x_hat = x_hat + torch.tensor(self.init_params["params"]["bias"].reshape(1,-1,1,1), dtype = x.dtype, device = x.device)
        
        return x_hat
        

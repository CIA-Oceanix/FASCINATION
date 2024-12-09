

import torch
import torch.nn as nn
import torch.nn.functional as F


            

class CNN_2D_Encoder(nn.Module):
    
    def __init__(self, 
                 input_channels: int,
                 base_channels: int,
                 latent_dim: int,
                 num_layers: int,
                 act_fn : object,
                 pooling_layer: object,
                 batch_norm: bool,
                 dropout_proba: float,
                 dtype: object):
        
        
        super().__init__()
        

        c_hid = base_channels

        layers = []
        
        layers.append([nn.Conv2d(input_channels, c_hid, kernel_size=1, padding=0, dtype = dtype)])
        
        for _ in range(num_layers):
            
            if batch_norm:
                batch_norm_layer = nn.BatchNorm2d(c_hid, dtype =dtype)

            else: 
                batch_norm_layer = nn.Identity()

            layers.append(
                [
                    batch_norm_layer,
                    act_fn,
                    pooling_layer,
                    nn.Conv2d(in_channels = c_hid, out_channels = c_hid*2, kernel_size= 1, padding= 0, dtype = dtype)
                ]
            )

            c_hid = c_hid*2
        
        layers.append(
            [
                nn.Flatten(),
                nn.Linear(in_features = c_hid * 240 * 240, out_features = 20, bias=True, dtype= dtype),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_proba, inplace=False),
                nn.Linear(in_features = 20, out_features = latent_dim, bias=True, dtype= dtype),
                act_fn
            ]
        )
        
        self.net = nn.Sequential(*sum(layers, []))
        
        
        
    def forward(self, x):
        return self.net(x)
    
    
    

class CNN_2D_Decoder(nn.Module):
    
    def __init__(self, 
                input_channels: int,
                base_channels: int,
                latent_dim: int,
                num_layers: int,
                act_fn: object,
                dtype: object):
        
        
        super().__init__()
        
        

        
        c_hid = 2*(num_layers -1)*base_channels
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, c_hid * 240 * 240, dtype = dtype),
            act_fn
        )
        
        
        
        layers = []
        
        for _ in range(num_layers-1):
            layers.append(
                [
                    nn.ConvTranspose2d(c_hid, c_hid//2, kernel_size=1, output_padding=0, padding=0, stride=1, dtype= dtype),
                    act_fn,
                    nn.Conv2d(c_hid//2, c_hid//2, kernel_size=1, padding=0, dtype= dtype),
                    act_fn,
                ]
            )
            
            c_hid = c_hid//2
            
        layers.append(
            [
                nn.ConvTranspose2d(c_hid, input_channels, kernel_size=1, output_padding=0, padding=0, stride=1),
                nn.Sigmoid(),
            ]
        )

        self.net = nn.Sequential(*sum(layers, []))
        
        
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 240, 240)
        x = self.net(x)
        return x
    
    


class AE_CNN_2D(nn.Module):
    
    def __init__(self, 
                input_channels: int = 107,
                base_channels: int = 100,
                latent_dim: int = 4,
                num_layers: int = 4,
                act_fn_str : str = "Sigmoid",
                pooling_str: str = "None",
                batch_norm: bool = False,
                dropout_proba: float = 0.,#*  https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html  Note that we do not apply Batch Normalization here. This is because we want the encoding of each image to be independent of all the other images. Otherwise, we might introduce correlations into the encoding or decoding that we do not want to have
                dtype_str: str = "float32"):
    

        super().__init__()
        

        pooling_dict = {"avg": nn.AvgPool2d(kernel_size=1, stride=1),
                        "max": nn.MaxPool2d(kernel_size=1, stride=1),
                        "None": nn.Identity()}
        
        act_fn_dict = {"Sigmoid": nn.Sigmoid(),
                       "Tanh": nn.Tanh(),
                       "Softmax": nn.Softmax(dim=1),
                       "Relu": nn.ReLU(),
                       "Gelu": nn.GELU(),
                       "None": nn.Identity()}
        
        
        self.model_dtype = getattr(torch, dtype_str)
        
                
        pooling_layer = pooling_dict[pooling_str]
        act_fn = act_fn_dict[act_fn_str]
        
        
        self.encoder = CNN_2D_Encoder(input_channels = input_channels,
                                      base_channels = base_channels,
                                      latent_dim = latent_dim,
                                      num_layers = num_layers,
                                      act_fn = act_fn,
                                      pooling_layer = pooling_layer,
                                      batch_norm = batch_norm,
                                      dropout_proba = dropout_proba,
                                      dtype = self.model_dtype
                                      )
        
        
        
        self.decoder = CNN_2D_Decoder(input_channels = input_channels,
                                      base_channels = base_channels,
                                      latent_dim =latent_dim,
                                      num_layers = num_layers,
                                      act_fn = act_fn,
                                      dtype = self.model_dtype
                                      )
        
    
    
    def forward(self, x):
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
        



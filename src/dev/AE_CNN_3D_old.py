

import torch
import torch.nn as nn
import torch.nn.functional as F


            

class CNN_3D_Encoder(nn.Module):
    
    def __init__(self, 
                 base_channels: int,
                 latent_dim: int,
                 num_layers: int,
                 spatial_dim: list,
                 act_fn : object,
                 batch_norm: bool,
                 dropout_proba: float,
                 linear_layer: bool,
                 dtype: object):
        
        
        super().__init__()


        self.base_channels = base_channels
        c_hid = base_channels
        channel_factor = 1
        ker = 3
        pad = 1
        layers = []
        

        
        for i in range(num_layers-1):
            
            if batch_norm:
                batch_norm_layer = nn.BatchNorm3D(c_hid, dtype =dtype)

            else: 
                batch_norm_layer = nn.Identity()
                
            if i >= num_layers-2:
                ker = 5

            layers.append(
                [   
                    nn.Conv3d(in_channels = c_hid, out_channels = c_hid*channel_factor,  kernel_size=(ker,1,1), padding=(pad,0,0), stride = 2, dtype = dtype),
                    batch_norm_layer,
                    act_fn,

                ]
            )

            c_hid = c_hid*channel_factor
            
        if linear_layer:
            layers.append(
                [
                    nn.Flatten(),
                    nn.Linear(in_features = c_hid *(spatial_dim[0]//(2**(num_layers-1))) * (spatial_dim[1]//(2**(num_layers-1))) * (spatial_dim[2]//(2**(num_layers-1))), out_features = latent_dim, bias=True, dtype= dtype),
                    # nn.ReLU(inplace=True),
                    # nn.Dropout(p=dropout_proba, inplace=False),
                    # nn.Linear(in_features = 20, out_features = latent_dim, bias=True, dtype= dtype),
                    act_fn
                ]
            )
            
        else:
            layers.append(
                [
                    nn.Conv3d(in_channels = c_hid, out_channels = latent_dim,  kernel_size=(ker,1,1), padding=(pad,0,0), stride = 2, dtype = dtype),
                    batch_norm_layer,
                    act_fn
                ]
            )

        
        self.net = nn.Sequential(*sum(layers, []))
        
        
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.base_channels, 1, 1, 1)
        return self.net(x)
    
    
    


class CNN_3D_Decoder(nn.Module):
    
    def __init__(self, 
                base_channels: int,
                latent_dim: int,
                num_layers: int,
                spatial_dim: object,
                act_fn: object,
                linear_layer: bool,
                dtype: object):
        
        
        super().__init__()
        
        
        self.base_channels = base_channels
        self.spatial_dim = spatial_dim
        self.num_layers = num_layers
        self.linear_layer = linear_layer
        c_hid = base_channels
        ker = 5
        pad_tr = (ker-3)//2   #* 1
        pad_conv = (ker - 1)//2  #*2

        layers = []
        
        if linear_layer:
            
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, c_hid * (spatial_dim[0]//(2**num_layers)) * (spatial_dim[1]//(2**num_layers)) * (spatial_dim[2]//(2**num_layers)), dtype = dtype),
                act_fn
            )
        
        else:
            layers.append(
                [
                    nn.ConvTranspose3d(latent_dim, c_hid, kernel_size=(ker,1,1), output_padding=(0,1,1), padding=(pad_tr,0,0), stride=2, dtype= dtype),  ##? output_padding=(2*(pad+1)-ker,1,1) 
                    act_fn,
                    nn.Conv3d(c_hid, c_hid, kernel_size=(ker,1,1), padding=(pad_conv,0,0), dtype= dtype),
                    act_fn,
                ]
            )
       
        

        
        for i in range(num_layers-1):
            
            if i >= num_layers-2:
                ker = 3
            
                pad_tr = (ker-3)//2   #*0
                pad_conv = (ker - 1)//2 #*1
            

            
            layers.append(
                [
                    nn.ConvTranspose3d(c_hid, c_hid, kernel_size=(ker,1,1), output_padding=(0,1,1), padding=(pad_tr,0,0), stride=2, dtype= dtype),  ##? output_padding=(2*(pad+1)-ker,1,1) 
                    act_fn,
                    nn.Conv3d(c_hid, c_hid, kernel_size=(ker,1,1), padding=(pad_conv,0,0), dtype= dtype),
                    act_fn,
                ]
            )
            
            c_hid = c_hid
            
        
        layers[-1][-1] = nn.AdaptiveMaxPool3d(output_size = spatial_dim)
        layers.append([nn.Sigmoid()])

        self.net = nn.Sequential(*sum(layers, []))
        
        
    def forward(self, x):
        
        if self.linear_layer:
            x = self.linear(x)
            x = x.reshape(x.shape[0], self.base_channels, (self.spatial_dim[0]//(2**self.num_layers)), (self.spatial_dim[1]//(2**self.num_layers)), (self.spatial_dim[2]//(2**self.num_layers)))
        x = self.net(x).squeeze(dim=1)
        
        return x
    
    


class AE_CNN_3D(nn.Module):
    
    def __init__(self, 
                base_channels: int = 1,
                latent_dim: int = 4,
                num_layers: int = 4,
                spatial_dim: list = [107, 240, 240],
                act_fn_str : str = "Sigmoid",
                batch_norm: bool = False,
                linear_layer: bool = True,
                dropout_proba: float = 0.,#*  https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html  Note that we do not apply Batch Normalization here. This is because we want the encoding of each image to be independent of all the other images. Otherwise, we might introduce correlations into the encoding or decoding that we do not want to have
                dtype_str: str = "float32"):
    

        super().__init__()
        

        
        act_fn_dict = {"Sigmoid": nn.Sigmoid(),
                       "Tanh": nn.Tanh(),
                       "Softmax": nn.Softmax(dim=1),
                       "Relu": nn.ReLU(),
                       "Gelu": nn.GELU(),
                       "None": nn.Identity()}
        
        
        self.model_dtype = getattr(torch, dtype_str)
        
                

        act_fn = act_fn_dict[act_fn_str]
        
        
        self.encoder = CNN_3D_Encoder(base_channels = base_channels,
                                      latent_dim = latent_dim,
                                      num_layers = num_layers,
                                      spatial_dim = spatial_dim,
                                      act_fn = act_fn,
                                      batch_norm = batch_norm,
                                      linear_layer = linear_layer,
                                      dropout_proba = dropout_proba,
                                      dtype = self.model_dtype
                                      )
        
        
        
        self.decoder = CNN_3D_Decoder(base_channels = base_channels,
                                      latent_dim =latent_dim,
                                      num_layers = num_layers,
                                      spatial_dim = spatial_dim,
                                      act_fn = act_fn,
                                      linear_layer = linear_layer,
                                      dtype = self.model_dtype
                                      )
        
    
    
    def forward(self, x):
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
        



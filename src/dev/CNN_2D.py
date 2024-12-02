

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dim_reduction(nn.Module):
    """
    [ Conv3d => BatchNorm (optional) => ReLU => Pooling (Avg or Max)] 
    """

    def __init__(self, in_ch: int, batch_norm: bool = True, avg_pool: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        

        if batch_norm:
            batch_norm_layer = nn.BatchNorm2d(in_ch, dtype = dtype)

        else: 
            batch_norm_layer = nn.Identity()
            
        if avg_pool:
            pool_layer = nn.AvgPool2d(kernel_size=1, stride=1)
        
        else:
            pool_layer = nn.MaxPool2d(kernel_size=1, stride=1)
      
        self.net = nn.Sequential(
            batch_norm_layer,
            nn.ReLU(inplace=True),
            pool_layer
            )
        
        

    def forward(self, x):
        return self.net(x)
    

class Last_Block(nn.Module):
    
    def __init__(self, in_features: int, 
                 num_classes: int, 
                 dtype: torch.dtype = torch.float32,
                 final_act_func_str: str = "Sigmoid"):
        super().__init__()


        if final_act_func_str == "Sigmoid":
            final_act_func = nn.Sigmoid()
            
        elif  final_act_func_str == "Softmax":
            final_act_func = nn.Softmax(dim=1)
            
        elif  final_act_func_str in ["None", None] :
            final_act_func = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = 20, bias=True, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(in_features = 20, out_features = num_classes, bias=True, dtype=dtype),
            final_act_func   
            )
        
    
    def forward(self, x):
        return self.net(x)
    
    

class CNN_2D(nn.Module):
    

    def __init__(self,
                 num_classes: int = 2,
                 in_ch: int = 107,
                 channels_start: int = 20,
                 num_layers: int = 4,
                 batch_norm: bool = True,
                 avg_pool: bool = False,
                 dtype_str: str = 'float32',
                 final_act_func_str: str = 'Sigmoid',
                 spatial_dim: list = [240,240]):    
    
        """
                num_classes: Number of output classes required
                num_layers: Number of layers in each side of U-net (default 5)
                channels_start: Number of channels in first layer (default 64)    
        """
        
        
        super().__init__()
        self.num_layers = num_layers
        self.model_dtype =  getattr(torch, dtype_str)
        self.in_ch = in_ch

        layers = [nn.Conv2d(in_ch, channels_start, kernel_size=1, padding=0, dtype = self.model_dtype)]
        
        self.channels = channels_start
        
        for i in range(num_layers -1):
        

            layers.append(Dim_reduction(in_ch = self.channels, batch_norm = batch_norm, avg_pool = avg_pool, dtype = self.model_dtype))
            layers.append(nn.Conv2d(in_channels = self.channels, out_channels = self.channels + 20, kernel_size= 1, padding= 0, dtype = self.model_dtype))
            self.channels = self.channels + 20
            
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(self.channels, dtype = self.model_dtype))
        
        if avg_pool:
            layers.append(nn.AdaptiveAvgPool2d(output_size = spatial_dim))
        
        else:
            layers.append(nn.AdaptiveMaxPool2d(output_size = spatial_dim))
          
        layers.append(nn.ReLU(inplace=True))  
            
        layers.append(Last_Block(in_features = self.channels, num_classes =num_classes, dtype = self.model_dtype, final_act_func_str = final_act_func_str))
        
        self.net = nn.Sequential(*layers)
        
        
    def forward(self, x):

        
        x = self.net[:-1](x)
                   
        x = self.net[-1](x.permute(0,2,3,1))
        
        x = x.permute(0,3,1,2).squeeze(dim=1)
        
        return x
            
        
        
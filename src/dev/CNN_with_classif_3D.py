

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Dim_reduction(nn.Module):
    """
    [ Conv3d => BatchNorm (optional) => ReLU => Pooling (Avg or Max)] 
    """

    def __init__(self, in_ch: int, batch_norm: bool = True, avg_pool: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        

        if batch_norm:
            batch_norm_layer = nn.BatchNorm3d(in_ch, dtype = dtype)

        else: 
            batch_norm_layer = nn.Identity()
            
        if avg_pool:
            pool_layer = nn.AvgPool3d(kernel_size=(2,1,1), stride=(2,1,1))
        
        else:
            pool_layer = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))
      
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
                 final_act_func_str = "Sigmoid"):
        super().__init__()
        

        if final_act_func_str == "Sigmoid":
            final_act_func = nn.Sigmoid()
            
        elif  final_act_func_str == "Softmax":
            final_act_func = nn.Softmax(dim=1)

        elif  final_act_func_str in ["None", None]:
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
    
    

class CNN_with_classif_3D(nn.Module):
    

    def __init__(self,
                 num_classes_pred: int = 1,
                 num_classes_classif : int = 2,
                 in_ch: int = 3,
                 channels_start: int = 20,
                 num_layers: int = 4,
                 batch_norm: bool = True,
                 avg_pool: bool = False,
                 dtype_str: str = 'float32',
                 spatial_dim: list = [240,240],
                 loss_weight: dict = {"no_ecs_weight": 1, "ecs_weight": 100}):    
    
        """
                num_classes: Number of output classes required
                num_layers: Number of layers in each side of U-net (default 5)
                channels_start: Number of channels in first layer (default 64)    
        """
        
        
        super().__init__()
        self.num_layers = num_layers
        self.model_dtype =  getattr(torch, dtype_str)
        self.in_ch = in_ch
        self.no_ecs_weight = np.float32(loss_weight["no_ecs_weight"])
        self.ecs_weight = np.float32(loss_weight["ecs_weight"])

        layers = [nn.Conv3d(in_ch, channels_start, kernel_size=(3, 1, 1), padding=(1,0,0), dtype = self.model_dtype)]
        
        self.channels = channels_start
        
        for _ in range(num_layers -1):
        

            layers.append(Dim_reduction(in_ch = self.channels, batch_norm = batch_norm, avg_pool = avg_pool, dtype = self.model_dtype))
            layers.append(nn.Conv3d(in_channels = self.channels, out_channels = self.channels + 20, kernel_size=(3, 1, 1), padding=(1,0,0), dtype = self.model_dtype))
            self.channels = self.channels + 20
            
        
        if batch_norm:
            layers.append(nn.BatchNorm3d(self.channels, dtype = self.model_dtype))
        
        if avg_pool:
            layers.append(nn.AdaptiveAvgPool3d(output_size = (1 , *spatial_dim)))
        
        else:
            layers.append(nn.AdaptiveMaxPool3d(output_size = (1, *spatial_dim)))
          
        layers.append(nn.ReLU(inplace=True))  
        
        layers.append(Last_Block(in_features = self.channels, num_classes = num_classes_pred, dtype = self.model_dtype, final_act_func_str = "Sigmoid"))  ##* prediction layer

        layers.append(Last_Block(in_features = self.channels, num_classes = num_classes_classif, dtype = self.model_dtype, final_act_func_str = None)) #Sigmoid if not BCE but NLLEloss ##* classification layer
        
        self.net = nn.Sequential(*layers)
        
        
    def forward(self, x):
        
        
        x = x.unsqueeze(1).repeat(1, self.in_ch, 1, 1, 1)
        
        x = self.net[:-2](x)
                   
        ecs_pred = self.net[-2](x.permute(0,2,3,4,1))
        ecs_classif = self.net[-1](x.permute(0,2,3,4,1))
        
        ecs_pred, ecs_classif = ecs_pred.permute(0,4,1,2,3).squeeze(dim=(1,2)), ecs_classif.permute(0,4,1,2,3).squeeze(dim=2)
        
        return ecs_pred, ecs_classif
        
        


import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet_2D(nn.Module):

    def __init__(
            self,
            num_classes: int,
            in_ch: int = 107,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            batch_norm: bool = True,
            avg_pool: bool = False,
            dtype_str: str = 'float32',
            final_act_func_str = "Sigmoid",
            output_spatial_shape: tuple = (240,240)
    ):
        """
        Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
        <https://arxiv.org/abs/1505.04597>`_

        Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_
            - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

        Args:
            num_classes: Number of output classes required
            num_layers: Number of layers in each side of U-net (default 5)
            features_start: Number of features in first layer (default 64)
            bilinear (bool): Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
        """
        super().__init__()
        self.num_layers = num_layers
        self.model_dtype =  getattr(torch, dtype_str)

        in_ch = 107


        layers = [DoubleConv(in_ch, features_start, batch_norm = batch_norm,  dtype = self.model_dtype)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2, avg_pool = avg_pool, dtype = self.model_dtype))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear, dtype = self.model_dtype))
            feats //= 2

        #layers.append(nn.Conv2d(feats, num_classes, kernel_size=(1,9,9) ))
        layers.append(nn.Conv2d(feats, num_classes,kernel_size=1))
        
        if avg_pool:
            layers.append(nn.AdaptiveAvgPool2d(output_size = output_spatial_shape))
        
        else:
            layers.append(nn.AdaptiveMaxPool2d(output_size = output_spatial_shape))
            
        
        if final_act_func_str == "Sigmoid":
            final_act_func = nn.Sigmoid()
            
        elif  final_act_func_str == "Softmax":
            final_act_func = nn.Softmax(dim=1)
            
        elif  final_act_func_str in ["None", None] :
            final_act_func = nn.Identity()

        layers.append(final_act_func)
        

        self.layers = nn.ModuleList(layers)



    def forward(self, x):
            
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-3]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        
        for layer in self.layers[-3:]:
            xi[-1] = layer(xi[-1])
            
        return xi[-1].squeeze(dim = 1)


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__()
        

        if batch_norm:
            batch_norm_layer = nn.BatchNorm2d(out_ch, dtype = dtype)

        else: 
            batch_norm_layer = nn.Identity()
            

        conv_layer_1 = nn.Conv2d(in_ch, out_ch, kernel_size= 3, padding=1, dtype = dtype)  ##? kernel to 3 or (3,1,1)
        conv_layer_2 = nn.Conv2d(out_ch, out_ch, kernel_size= 3, padding=1, dtype = dtype)
        

            
                        
        self.net = nn.Sequential(
            conv_layer_1,
            batch_norm_layer,
            nn.ReLU(inplace=True),
            conv_layer_2,
            batch_norm_layer,
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int, avg_pool: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        

        if avg_pool:
            pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        
        else:
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                    

            
        self.net = nn.Sequential(
            pool_layer,
            DoubleConv(in_ch, out_ch, dtype = dtype)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.upsample = None

        

        conv_layer = nn.Conv2d(in_ch, in_ch // 2, kernel_size= 1, dtype = dtype)
            
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                conv_layer
                )
            
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2, dtype = dtype)
            
            



        self.conv = DoubleConv(in_ch, out_ch,  dtype = dtype)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2

        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
            



        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
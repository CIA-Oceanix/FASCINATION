

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(
            self,
            num_classes: int,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            three_dim : bool = False,
            batch_norm: bool = True,
            avg_pool: bool = False,
            dtype_str: str = 'float32',
            output_spatial_shape: tuple = (107,240,240)
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
        self.three_dim = three_dim
        
        if three_dim:
            in_ch = 3

        else:
            in_ch = 107   ##? can it be automatized 

        layers = [DoubleConv(in_ch, features_start, batch_norm = batch_norm, conv_3D = three_dim, dtype = self.model_dtype)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2, avg_pool = avg_pool, pool_3D = three_dim, dtype = self.model_dtype))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear, conv_3D = three_dim, dtype = self.model_dtype))
            feats //= 2

        #layers.append(nn.Conv3d(feats, num_classes, kernel_size=(1,9,9) ))
        layers.append(nn.Conv3d(feats, num_classes,kernel_size=1))
        
        if avg_pool:
            layers.append(nn.AdaptiveAvgPool3d(output_size = output_spatial_shape))
        
        else:
            layers.append(nn.AdaptiveMaxPool3d(output_size = output_spatial_shape))
            
        
        layers.append(torch.nn.LogSoftmax(dim=1))
        

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        
        if self.three_dim:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, batch_norm: bool = True, conv_3D: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        

        if batch_norm:
            if conv_3D: 
                batch_norm_layer = nn.BatchNorm3d(out_ch, dtype = dtype)
            else:
                batch_norm_layer = nn.BatchNorm2d(out_ch, dtype = dtype)
        else: 
            batch_norm_layer = nn.Identity()
            
        if conv_3D: 
            conv_layer_1 = nn.Conv3d(in_ch, out_ch, kernel_size=(3, 1, 1), padding=1, dtype = dtype)  ##? kernel to 3 or (3,1,1)
            conv_layer_2 = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), padding=1, dtype = dtype)
        
        else: 
            conv_layer_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dtype = dtype)
            conv_layer_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, dtype = dtype)
            
                        
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

    def __init__(self, in_ch: int, out_ch: int, avg_pool: bool = False, pool_3D: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        
        if pool_3D:
            if avg_pool:
                pool_layer = nn.AvgPool3d(kernel_size=2, stride=2)
            
            else:
                pool_layer = nn.MaxPool3d(kernel_size=2, stride=2)
                    
        else:
            if avg_pool:
                pool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
            
            else:
                pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.net = nn.Sequential(
            pool_layer,
            DoubleConv(in_ch, out_ch, conv_3D = pool_3D, dtype = dtype)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False, conv_3D: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.upsample = None
        self.conv_3D = conv_3D
        
        if conv_3D:
            conv_layer = nn.Conv3d(in_ch, in_ch // 2, kernel_size=(1, 1, 1), dtype = dtype)
            
            if bilinear:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    conv_layer
                    )
                
            else:
                self.upsample = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2, dtype = dtype)
                
            
        else:
            conv_layer = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, dtype = dtype)
            
            if bilinear:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    conv_layer
                    )
            
            else:
                self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2, dtype = dtype)
            


        self.conv = DoubleConv(in_ch, out_ch, conv_3D = conv_3D, dtype = dtype)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        if self.conv_3D:
            diff_d = x2.shape[2] - x1.shape[2]
            diff_h = x2.shape[3] - x1.shape[3]
            diff_w = x2.shape[4] - x1.shape[4]
            
            x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2, diff_d // 2, diff_d - diff_d // 2])
            
        else:
            diff_h = x2.shape[2] - x1.shape[2]
            diff_w = x2.shape[3] - x1.shape[3]
            
            x1 = F.pad(x1, [diff_h // 2, diff_h - diff_h // 2, diff_w // 2, diff_w - diff_w // 2])


        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
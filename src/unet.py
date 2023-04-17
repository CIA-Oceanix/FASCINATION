#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:28:35 2023

@author: lucarou
"""
""" U-Net model from Weyn, Jonathan A., Dale R. Durran, and Rich Caruana. "Improving data‐driven global weather prediction using deep convolutional neural networks on a cubed sphere." 
    Journal of Advances in Modeling Earth Systems 12.9 (2020): e2020MS002109.

    This is a pytorch recreation of the model used in their paper, dedicated to the prediction of oceanography related variables.
    Each Conv2D layer except the last one is followed by a LeakyReLU10 (LeakyReLU capped at 10). No Dropout, no BatchNorm layers, only AveragePooling2D.
    We are not applying the cubed sphere method described in the paper.
    """

import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl

padding_mode = 'reflect'


class ModifiedReLU(nn.Module):
    def __init__(self):
        super(ModifiedReLU, self).__init__()

    def forward(self, x):
        return torch.where(x <= 0, 0.1 * x, torch.where(x >= 10, 10, x))
    
    
class DoubleConv(nn.Module):
    """(Convolution => ModifiedReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                ModifiedReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
                ModifiedReLU()
            )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
class UNet(pl.LightningModule):
    def __init__(self, n_channels, n_var, time_steps=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_var = n_var
        self.bilinear = bilinear
        self.time_steps = time_steps

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DoubleConv(64, 128, 64)
        )
        self.up1 = Up(64, 64)
        self.doubleconv1 = DoubleConv(128, 64, 32)
        self.up2 = Up(32, 32)
        self.doubleconv2 = DoubleConv(64, 32)
        self.outc = OutConv(32, n_var*time_steps)

    def forward(self, x):
        x = self.inc(x)
        x1= self.down1(x)
        x2 = self.down2(x1)

        x = self.up1(x2)
        x = torch.cat([x, x2], dim = 1)
        x = self.doubleconv1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim = 1)
        x = self.doubleconv2(x)
        out = self.outc(x)
        
        return out
    
    def configure_optimizers(self):         #vérifier les dimensions de nos données, ça va dépendre de comment je les charge dans le dataloader
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx, seq_len):
        #assuming x = [x(t-delta_t), x(t)], and y = [y(t+delta_t), y(t+2*delta_t)]
        x, y = batch
        loss = 0
        for i in range(seq_len):
            y_hat = self(x[i])
            loss += nn.MSELoss()(y_hat, y[i])
        loss /= seq_len
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, seq_len):
        #assuming x = [x(t-delta_t), x(t)], and y = [y(t+delta_t), y(t+2*delta_t)]
        x, y = batch
        loss = 0
        for i in range(seq_len):
            y_hat = self(x[i])
            loss += nn.MSELoss()(y_hat, y[i])
        loss /= seq_len
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx, seq_len): # rajouter l'ACC et éventuellement les métriques spectrales de hugo
        #assuming x = [x(t-delta_t), x(t)], and y = [y(t+delta_t), y(t+2*delta_t)]
        x, y = batch
        loss = 0
        for i in range(seq_len):
            y_hat = self(x[i])
            loss += torch.sqrt(nn.MSELoss()(y_hat, y[i]))
        loss /= seq_len
        self.log('test loss', loss)

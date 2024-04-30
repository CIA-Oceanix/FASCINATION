#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:45:48 2023

@author: lucarou
"""
""" Sound variable predictor based on the reconstruction from the auto-encoder. Using this neural network to guide the training of the auto-encoder so that
    it accurately reconstructs the variables of interest.
    """

from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import xarray as xr
import pandas as pd
from src.utils import psd_based_scores, rmse_based_scores


class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size= 3, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)
    
class ReduceDomain(pl.LightningModule):
    def __init__(self, resize_factor): # assuming the domain is a square
        super().__init__()
        self.resize_factor = resize_factor
        self.block = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.block(x)

class IncreaseDomain(pl.LightningModule):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode='bicubic', align_corners=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.block(x)


class AcousticPredictor(pl.LightningModule):
    def __init__(self, input_depth, acoustic_variables=2, lr=1e-3, T_max=10):
        super(AcousticPredictor, self).__init__()
        self.lr = lr
        self.T_max = T_max
        self.test_data = None

        self.conv1 = ConvBlock(input_depth, 96)
        self.conv2 = ConvBlock(96, 64)
        self.conv3 = ConvBlock(64, 32)
        self.conv4 = ConvBlock(32, 16)
        self.finalconv = ConvBlock(16, acoustic_variables)

    def forward(self, x):
        return self.finalconv(
                    self.conv4 (
                        self.conv3(
                            self.conv2(
                                self.conv1(x)
                            )
                        )
                    )
                )
        
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.T_max)
        return [self.optimizer], [self.scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.sqrt(nn.MSELoss()(y[:,:2,:,:], output))
        self.log('train_loss', loss, on_step= False, on_epoch=True)
        return loss
    
    # def on_train_epoch_end(self, outputs):
    #     self.scheduler.step()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.MSELoss()(output, y[:,:2,:,:])
        self.log('val_loss', loss, on_step= False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        x, y = batch
        output = self(x)
        y_split, output_split = torch.split(y, 1, dim=1), torch.split(output, 1, dim=1)
        self.test_data.append(torch.stack([y[:,:2,:,:], output], dim=1))
        test_loss = {
            "cutoff_freq": 0.0,
            "ecs": 0.0
        }
        test_loss["cutoff_freq"] = torch.sqrt(nn.MSELoss()(y_split[0]*10000, output_split[0]*10000))
        test_loss["ecs"] = torch.sqrt(nn.MSELoss()(y_split[1]*670.25141631, output_split[1]*670.25141631))
        self.log_dict(test_loss, on_step= False, on_epoch=True)
        return test_loss
    
    
    
if __name__ == '__main__':
    checkpoint_path = '/homes/o23gauvr/Documents/thèse/code/FASCINATION/outputs/accoustic_predictor/2024-04-04_15-22/checkpoints/val_loss=0.02-epoch=970.ckpt'
    input_depth = 107
    model = AcousticPredictor.load_from_checkpoint(checkpoint_path = checkpoint_path,input_depth = input_depth)
    
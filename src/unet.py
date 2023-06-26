#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:28:35 2023

@author: lucarou
"""
""" U-Net model from Weyn, Jonathan A., Dale R. Durran, and Rich Caruana. "Improving data‐driven global weather prediction using deep convolutional neural networks on a cubed sphere." 
    Journal of Advances in Modeling Earth Systems 12.9 (2020): e2020MS002109.

    This is a pytorch implementation of the model described in their paper, dedicated to the prediction of oceanography related variables.
    Each Conv2D layer except the last one is followed by a LeakyReLU10 (LeakyReLU capped at 10). No Dropout, no BatchNorm layers, only AveragePooling2D.
    We are not applying the cubed sphere method described in the paper.
    """

from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
import xarray as xr
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utils import psd_based_scores, rmse_based_scores

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

    def __init__(self, in_channels, out_channels, bilinear=False):
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
    def __init__(self, n_var, io_time_steps=2, integration_steps=2, loss_by_step=1):
        super(UNet, self).__init__()
        self.n_var = n_var
        self.io_time_steps = io_time_steps
        self.integration_steps = integration_steps
        self.loss_by_step = loss_by_step

        self.test_outputs_gt = {
            "outputs": [],
            "gt": []
        }

        self.inc = DoubleConv(n_var*io_time_steps, 32)
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DoubleConv(64, 64, 128)
        )
        self.up1 = Up(64, 64)
        self.doubleconv1 = DoubleConv(128, 32, 64)
        self.up2 = Up(32, 32)
        self.doubleconv2 = DoubleConv(64, 32)
        self.outc = OutConv(32, n_var*io_time_steps)

    def forward(self, x):
        # down
        x0 = self.inc(x)
        x1= self.down1(x0)
        x2 = self.down2(x1)
        # up
        x = self.up1(x2)
        x = torch.cat([x, x1], dim = 1)
        x = self.doubleconv1(x)
        x = self.up2(x)
        x = torch.cat([x, x0], dim = 1)
        x = self.doubleconv2(x)
        out = self.outc(x)
        
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # calculating loss on integration_steps forward passes to force stability of auto-regressive process
        x, y = batch
        outputs = [self(x)]
        for i in range(1, self.integration_steps):
            x0 = outputs[i-1]
            outputs.append(self(x0))
        outputs = torch.stack(outputs).view(*y.shape)
        loss = nn.MSELoss()(outputs, y)/self.integration_steps
        self.log('train_loss', loss, on_step= False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = [self(x)]
        for i in range(1, self.integration_steps):
            x0 = outputs[i-1]
            outputs.append(self(x0))
        outputs = torch.stack(outputs).view(*y.shape)
        loss = nn.MSELoss()(outputs, y)/self.integration_steps
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx): # ACC ?
        x, y = batch
        outputs = [self(x)]
        for i in range(1, self.integration_steps):
            x0 = outputs[i-1]
            outputs.append(self(x0))
        outputs = torch.stack(outputs).view(*y.shape) 
        RMSE = torch.sqrt(nn.MSELoss()(outputs, y)/self.integration_steps)
        self.log('RMSE', RMSE, on_step= True, on_epoch=True)

        return outputs
    
    def on_test_batch_end(self, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int):
        self.test_outputs_gt["outputs"].append(outputs)
        self.test_outputs_gt["gt"].append(batch.tgt)
    
    def on_test_end(self):
        outputs_tensor = self.test_outputs_gt["outputs"].pop(0)
        gt_tensor = self.test_outputs_gt["gt"].pop(0)
        while len(self.test_outputs_gt["outputs"]) > 0 and len(self.test_outputs_gt["gt"]) > 0:
            outputs_tensor = torch.cat((outputs_tensor, self.test_outputs_gt["outputs"].pop(0)), dim=0)  # il y aura peut-être une dimension de batch en plus quelque part, faire attention lors des tests
            gt_tensor = torch.cat((gt_tensor, self.test_outputs_gt["gt"].pop(0)), dim=0)
        dm = self.trainer.datamodule
        time, var, lat, lon = dm.test_time, dm.test_var, dm.test_lat, dm.test_lon
        outputs_array = xr.DataArray(outputs_tensor, coords=[time, var, lat, lon], dims=['time', 'var', 'lat', 'lon'])
        gt_array = xr.DataArray(gt_tensor, coords=[time, var, lat, lon], dims=['time', 'var', 'lat', 'lon'])
        metrics = {
            **dict(
            zip(
                ["λx", "λt"],
                psd_based_scores(outputs_array, gt_array)[1:]
                )
            ),
            **dict(
            zip(
                ["μ", "σ"],
                rmse_based_scores(outputs_array, gt_array)[2:],
                )
            ),
        }
        return pd.Series(metrics, name="osse_metrics")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:57:22 2023

@author: lucarou
"""
""" New network implementation for the FASCINATION project: Convolutional Auto-Encoder. The goal now is not to predict the soud speed conditions in the ocean in the future, but rather to learn how to compress them as much as possible,
    for 3D + t data, and to reconstruct it as good as possible with respect to variables deducted from the sound profiles.
    """

from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
import xarray as xr
import pandas as pd
import hydra
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utils import psd_based_scores, rmse_based_scores

#version 0 de la loss où on prend en compte uniquement la loss de reconstruction avec une MSE, on regarde ni les gradients verticaux ni les métriques de 
#reconstruction des paramètres acoustiques

class Autoencoder(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(Autoencoder, self).__init__()
        self.lr = lr
        self.encoder = nn.Sequential(
            nn.Conv2d(240, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 240, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.MSELoss(output, y)
        self.log('train_loss', loss, on_step= False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.MSELoss(output, y)
        self.log('val_loss', loss, on_step= False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.MSELoss(output, y)
        self.log('test_loss', loss, on_step= False, on_epoch=True)
        return loss

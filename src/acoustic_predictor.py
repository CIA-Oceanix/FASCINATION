#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:45:48 2023

@author: lucarou
"""
""" Sound variable predictor based on the reconstruction from the auto-encoder. Using this neural network to guide the training of the auto-encoder so that
    it accurately reconstructs the variables of interest.
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

class AcousticPredictor(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(AcousticPredictor, self).__init__()
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

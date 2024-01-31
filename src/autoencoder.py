#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:57:22 2023

@author: lucarou
"""
""" New network implementation for the FASCINATION project: Convolutional Auto-Encoder. The goal now is not to predict the soud speed conditions in the ocean in the future, but rather to learn how to compress them as much as possible,
    for 3D + t data, and to reconstruct it as good as possible with respect to variables deducted from the sound profiles.
    """

import torch
import torch.nn as nn
import pytorch_lightning as pl

class AutoEncoder(pl.LightningModule):
    def __init__(self, lr=1e-3, acoustic_predictor=None):
        super(AutoEncoder, self).__init__()
        self.lr = lr
        self.test_data = None

        self.acoustic_predictor = acoustic_predictor
        if self.acoustic_predictor != None:
            self.acoustic_predictor.eval()

        self.encoder = nn.Sequential(
            nn.Conv2d(107, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 107, kernel_size=2, stride=2),
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
        loss = torch.sqrt(nn.MSELoss()(output, x))
        if self.acoustic_predictor != None:
            loss = loss + torch.sqrt(nn.MSELoss()(self.acoustic_predictor(output), y))
        self.log('train_loss', loss, on_step= True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.sqrt(nn.MSELoss()(output, x))
        self.log('val_loss', loss, on_step= False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        x, y = batch
        output = self(x)
        self.test_data.append(torch.stack([x, output], dim=1))
        loss = torch.sqrt(nn.MSELoss()(output*(1552.54994512 - 1438)+1438, x*(1552.54994512 - 1438)+1438))
        self.log('test_loss', loss, on_step= False, on_epoch=True)
        return loss

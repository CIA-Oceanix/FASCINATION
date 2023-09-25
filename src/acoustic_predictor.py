#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:45:48 2023

@author: lucarou
"""
""" Sound variable predictor based on the reconstruction from the auto-encoder. Using this neural network to guide the training of the auto-encoder so that
    it accurately reconstructs the variables of interest.
    """

import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size= 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class AcousticPredictor(pl.LightningModule):
    def __init__(self, input_depth, acoustic_variables=2, lr=1e-3):
        super(AcousticPredictor, self).__init__()
        self.lr = lr

        self.conv1 = ConvBlock(input_depth, 64)
        self.conv2 = ConvBlock(64, 32)
        self.conv3 = ConvBlock(32, 16)
        self.out = ConvBlock(16, acoustic_variables)

    def forward(self, x):
        return self.out(
            self.conv3(
                self.conv2(
                    self.conv1(x)
                )
            )
        )
    
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

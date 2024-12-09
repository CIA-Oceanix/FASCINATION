#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


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
    

class AcousticPredictor(pl.LightningModule):
    def __init__(self, input_depth, opt_fn, acoustic_variables=1):
        super(AcousticPredictor, self).__init__()
        
        self.opt_fn = opt_fn

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
        return self.opt_fn(self)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)


    def cosanneal_lr_adamw(self, lr, T_max, weight_decay=0.):
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay= weight_decay)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=T_max,
            ),
        }

    def adamw(self,lr):
        return  torch.optim.AdamW(self.parameters(), lr=lr)


    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]
    

    def step(self, batch, phase =""):
        # if self.training and batch.isfinite().float().mean() < 0.9:
        #     return None, None
        ##TODO: manage this test
        
        loss, out = F.mse_loss(out,batch)  ##! for conv2D the patch should be of size 1 and we sueeze the tensor
        
        with torch.no_grad():
            #self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_mse", loss, prog_bar=True, on_step=True, on_epoch=True)
        

        return loss, out
        

    
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
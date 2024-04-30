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
from torchsummary import summary

class AutoEncoder(pl.LightningModule):
    def __init__(self,x_min = 1438, x_max = 1552.54994512, lr=1e-3, arch_shape = "16_60",final_act_func = 'sigmoid',  acoustic_predictor=None, accoustic_training = False):
        super(AutoEncoder, self).__init__()
        self.lr = lr
        self.test_data = None
        self.final_act_func = final_act_func
        self.acoustic_predictor = acoustic_predictor
        self.acoustic_predictor.eval()
        self.accoustic_training = accoustic_training

        self.architecture(arch_shape)
        
        self.x_min = x_min
        self.x_max = x_max
        
        # if torch.cuda.is_available():
        #     self.input_da.to('cuda')
        # self.log_model_summary()
        


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        ##TODO use Adamw
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.sqrt(nn.MSELoss()(output, x))
        self.log('train_loss_AE', loss, on_step= True, on_epoch=True)
        ecs_rmse = torch.sqrt(nn.MSELoss()(self.acoustic_predictor(output)[:,-1,:,:], y[:,1,:,:]))

            #cut_off_rmse = = torch.sqrt(nn.MSELoss()(self.acoustic_predictor(output)[:,1,:,:], y[:,1,:,:]))
            #loss = loss + torch.sqrt(nn.MSELoss()(self.acoustic_predictor(output), y[:,:2,:,:]))
        if self.accoustic_training != False:
                loss = loss + ecs_rmse
        self.log('train_loss', loss, on_step= True, on_epoch=True)
        self.log('train_ECS_rmse', ecs_rmse, on_step= True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = torch.sqrt(nn.MSELoss()(output, x))
        ecs_rmse = torch.sqrt(nn.MSELoss()(self.acoustic_predictor(output)[:,1,:,:], y[:,1,:,:]))
        
        self.log('val_loss', loss, on_step= False, on_epoch=True)
        self.log('val_ECS_rmse', ecs_rmse, on_step= True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        x, y = batch
        output = self(x)
        self.test_data.append(torch.stack([x, output], dim=1))
        loss = torch.sqrt(nn.MSELoss()(output*(self.x_max - self.x_min)+self.x_min, x*(self.x_max - self.x_min)+self.x_min))        
        ecs_rmse = torch.sqrt(nn.MSELoss()(self.acoustic_predictor(output)[:,1,:,:], y[:,1,:,:]))
        
        self.log('test_ecs_rmse_normalized', ecs_rmse, on_step= False, on_epoch=True)
        self.log('test_loss', loss, on_step= False, on_epoch=True)

        return loss



    def architecture(self, arch_shape):
        
        if self.final_act_func == 'sigmoid':
            final_act_func = nn.Sigmoid()
        elif self.final_act_func == 'relu':
            final_act_func = nn.ReLU()
        
        if arch_shape == "32_120": 
            
            self.encoder = nn.Sequential(
                nn.Conv2d(107, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # nn.Conv2d(64, 32, kernel_size=3, padding=1),
                # nn.ReLU(),
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 107, kernel_size=2, stride=2),
                nn.Sigmoid()
            )
                    
                    
        if arch_shape == "16_60": 
            
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
            
        if arch_shape == "8_30": 
            
            self.encoder = nn.Sequential(
                nn.Conv2d(107, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 107, kernel_size=2, stride=2),
                nn.Sigmoid()
            )
            
        # if arch_shape == "4_15": 
        #     ###TODO: put batchnorm
        #     ###TODO check diminution kernel dans convo
        #     self.encoder = nn.Sequential(
        #         nn.Conv2d(107, 64, kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(32, 16, kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(16, 8, kernel_size=3, padding=1),
        #         nn.ReLU(),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(8, 4, kernel_size=3, padding=1),
        #         nn.ReLU() 
        #     )
            
        #     ###TODO: enlever ReLU
        #     ###TODO check stride
        #     self.decoder = nn.Sequential(
        #         nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(16, 64, kernel_size=2, stride=2),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(64, 107, kernel_size=2, stride=2),
        #         nn.Sigmoid() ###! sortie [0;1] ? test softplus, htgt, relu
        #     )
            
                      
        
        
        if arch_shape == "4_15": 
            ###TODO: put batchnorm
            ###TODO check diminution kernel dans convo
            self.encoder = nn.Sequential(
                nn.Conv2d(107, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(8, 4, kernel_size=3, padding=1)
                #nn.ReLU()
                
            )
            ###TODO: enlever ReLU
            ###TODO check stride
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2),
                #nn.ReLU(),
                nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
                #nn.ReLU(),
                nn.ConvTranspose2d(16, 64, kernel_size=2, stride=2),
                #nn.ReLU(),
                nn.ConvTranspose2d(64, 107, kernel_size=2, stride=2),
                final_act_func ###* sigmoid ok si normalization, valeur entre 0 et 1
            )
            
        if arch_shape == "4_4": 
            ###TODO: put batchnorm
            ###TODO check diminution kernel dans convo
            self.encoder = nn.Sequential(
                nn.Conv2d(107, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(8, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=3)
                #nn.ReLU()
                
            )
            ###TODO: enlever ReLU
            ###TODO check stride
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(4, 4, kernel_size=4, stride=3, output_padding=2), ##! check if output_padding makes sense
                nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2),
                #nn.ReLU(),
                nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
                #nn.ReLU(),
                nn.ConvTranspose2d(16, 64, kernel_size=2, stride=2),
                #nn.ReLU(),
                nn.ConvTranspose2d(64, 107, kernel_size=2, stride=2),
                final_act_func ###* sigmoid ok si normalization, valeur entre 0 et 1
            )
            
        if arch_shape == "no_pool_4" :
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=107, out_channels=64, kernel_size=1, stride=1, padding=0),  # Conv1
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0), # Conv2
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0), # Conv3
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1, stride=1, padding=0), # Conv4
                nn.ReLU(inplace=True)
            )
            # Decoder layers
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0, output_padding=0), # ConvTranspose1
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, output_padding=0), # ConvTranspose2
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, output_padding=0), # ConvTranspose3
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=64, out_channels=107, kernel_size=1, stride=1, padding=0, output_padding=0), # ConvTranspose4
                nn.Sigmoid()  # Output activation function
            )
    
        if arch_shape == "pca_4" :
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=107, out_channels=4, kernel_size=1, stride=1, padding=0)  # Conv1
            )
            # Decoder layers
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=4, out_channels=107, kernel_size=1, stride=1, padding=0, output_padding=0), # ConvTranspose1
                final_act_func # Output activation function
            )
            
        if arch_shape == "pca_50" :
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=107, out_channels=50, kernel_size=1, stride=1, padding=0)  # Conv1
            )
            # Decoder layers
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=50, out_channels=107, kernel_size=1, stride=1, padding=0, output_padding=0), # ConvTranspose1
                final_act_func # Output activation function
            )
            
        if arch_shape == "pca_107" :
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=107, out_channels=107, kernel_size=1, stride=1, padding=0)  # Conv1
            )
            # Decoder layers
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=107, out_channels=107, kernel_size=1, stride=1, padding=0, output_padding=0), # ConvTranspose1
                final_act_func # Output activation function
            )
            
    def log_model_summary(self):
        self.log(summary(self,input_size = (107,240,240)))
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
import numpy as np
#from UNet_3D import UNet


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
    def __init__(self, input_depth = 107, acoustic_variables=1, lr=1e-3, T_max=10, classif_weight = 1, pred_weight = 1, kernel_size = 8, arch_shape = "dense_3D_CNN_ReLu", dtype_str = 'float32'):
        super(AcousticPredictor, self).__init__()
        self.lr = lr
        self.T_max = T_max
        self.test_data = None
        self.classif_weight = classif_weight
        self.pred_weight = pred_weight
        self.arch_shape = arch_shape
        self.kernel_size = kernel_size
        self.acoustic_variables = acoustic_variables
        self.input_depth = input_depth
        self.model_dtype = getattr(torch, dtype_str)
        self.architecure()
        
        self.save_hyperparameters()
    


    def forward(self, x):
        
        if "3D" in self.arch_shape:
            x = x.unsqueeze(1)
        
        encoded = self.encoder(x)
        
        if self.arch_shape == 'dense_3D_CNN_ReLu':
            encoded = encoded.squeeze(dim=1).permute(0,2,3,1)  ##* nn.linear need channels/features at the end
            ecs_pred = self.ecs_pred_model(encoded).permute(0,3,1,2)
            ecs_classif = self.ecs_classif_model(encoded).permute(0,3,1,2)  ##* put the channels back in place after the batchsize
            return ecs_pred, ecs_classif
        
        if "3D" in self.arch_shape:
            encoded.squeeze(dim=1)
            
        return encoded
    
    
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.T_max)
        return [self.optimizer], [self.scheduler]
    
    
    
    def training_step(self, batch, batch_idx):

        return self.step(batch,'train')

        # else:   
        #     x, y = batch               
        #     output = self(x)  ##* does not require squeeze() even with 3D arch
        #     loss = torch.sqrt(nn.MSELoss()(y, output))
        #     self.log('train_loss', loss, on_step= False, on_epoch=True)
        #     return loss
    
    # def on_train_epoch_end(self, outputs):
    #     self.scheduler.step()
    
    def validation_step(self, batch, batch_idx):

        return self.step(batch,'val')
    
        # else:
        #     x, y = batch
        #     output = self(x)
        #     loss = nn.MSELoss()(output, y)
        #     self.log('val_loss', loss, on_step= False, on_epoch=True)
        #     return loss
        
    # def on_validation_epoch_end(self):
         
    #     val_loss = self.trainer.callback_metrics['val_loss']
    #     print('\n',val_loss, val_loss.item())
    #     print(self.trainer.early_stopping_callback.best_score,self.trainer.early_stopping_callback.best_score.item())
    #     print(val_loss.item() - self.trainer.early_stopping_callback.best_score.item())
    #     print(torch.lt(val_loss - 1.0e-6,self.trainer.early_stopping_callback.best_score))
        
        
    def test_step(self, batch, batch_idx):
        
        pred_loss, classif_loss = self.step(batch,'test')
        
        
        ecs_rmse = torch.sqrt(pred_loss)*670.25141631
        self.log("ECS RMSE", ecs_rmse)
        self.log("ECS classification corss entropy loss",classif_loss)
        
        
        # if batch_idx == 0:
        #     self.test_data = []
        
        # else:
        #     x, y = batch
        #     output = self(x)
        #     #y_split, output_split = torch.split(y, 1, dim=1), torch.split(output, 1, dim=1)
        #     #self.test_data.append(torch.cat([y, output], dim=1))
        #     test_loss = {
        #         "ecs": 0.0
        #     }   ##add 'cutoff_freq if needed
        #     #test_loss["cutoff_freq"] = torch.sqrt(nn.MSELoss()(y_split*10000, output_split*10000))
        #     test_loss["ecs"] = torch.sqrt(nn.MSELoss()(y, output)).item()*670.25141631
        #     self.log_dict(test_loss, on_step= False, on_epoch=True)
        #     return test_loss
    
    
    def step(self, batch, phase = ""):
        
        x,y = batch
        
        if self.arch_shape == 'dense_3D_CNN_ReLu':
            y_classif = y.squeeze(dim=1).long()
            y_classif[y_classif !=0] = 1
            ecs_pred, ecs_classif = self(x)
            pred_loss = nn.MSELoss()(ecs_pred,y)
            classif_loss = nn.CrossEntropyLoss()(ecs_classif,y_classif) ##* a softmax is applied inside the loss function
            full_loss = self.pred_weight*pred_loss + self.classif_weight*classif_loss
            
        else:
            ecs_pred = self(x)
            classif_loss = np.nan
            pred_loss = nn.MSELoss()(ecs_pred,y)
            full_loss = pred_loss 

        if phase == 'test':
            return pred_loss, classif_loss


        
        else:
 
            # loss_dict = {f"prediction loss {phase}": pred_loss,
            #             f"classification loss {phase}": classif_loss,
            #             f"{phase}_loss": full_loss}
            
            # self.log_dict(loss_dict, prog_bar=True, on_step=None, on_epoch=True)
            
            self.log(f"prediction loss {phase}", pred_loss,  prog_bar=False, on_step=None, on_epoch=True)
            self.log(f"classification loss {phase}", classif_loss,  prog_bar=False, on_step=None, on_epoch=True)
            self.log(f"{phase}_loss", full_loss,  prog_bar=True, on_step=None, on_epoch=True)
            
            #self.log(f"{phase}_loss",full_loss)
            
            return full_loss
        
        
        
    
    def architecure(self):
        
        if self.arch_shape == "lucas_model":
            
            # self.conv1 = ConvBlock(self.input_depth, 96)
            # self.conv2 = ConvBlock(96, 64)
            # self.conv3 = ConvBlock(64, 32)
            # self.conv4 = ConvBlock(32, 16)
            # self.finalconv = ConvBlock(16, self.acoustic_variables)
            
            self.encoder = nn.Sequential(
                            ConvBlock(self.input_depth, 96),
                            ConvBlock(96, 64),
                            ConvBlock(64, 32),
                            ConvBlock(32, 16),
                            ConvBlock(16, self.acoustic_variables)
                            )
                    
        elif self.arch_shape == 'dense_2D_CNN_ReLu':
            self.encoder = nn.Sequential(
                                nn.Conv2d(in_channels=self.input_depth, out_channels=64, kernel_size=1, stride=1, padding=0),  # Conv1
                                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0), # Conv2
                                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0), # Conv3
                                nn.Conv2d(in_channels=16, out_channels=self.acoustic_variables, kernel_size=1, stride=1, padding=0), # Conv4
                                nn.ReLU()
                            )
            
        # elif self.arch_shape == 'dense CNN Softmax':
        #     self.encoder = nn.Sequential(
        #                         nn.Conv2d(in_channels=self.input_depth, out_channels=64, kernel_size=1, stride=1, padding=0),  # Conv1
        #                         nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0), # Conv2
        #                         nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0), # Conv3
        #                         nn.Conv2d(in_channels=16, out_channels=self.acoustic_variables, kernel_size=1, stride=1, padding=0), # Conv4
        #                         nn.Softmax()
        #                     )
        
        elif self.arch_shape == 'dense_3D_CNN_ReLu_1st_kernel_size':
            
            depth_kernel_size = self.get_n_th_smallest_denominator(1)
            
            num_layers = int((self.input_depth-1)/(depth_kernel_size-1))
            conv_layers = [nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(depth_kernel_size, 1, 1), stride=1, padding=0) for _ in range(num_layers)]
            self.encoder = nn.Sequential(*conv_layers,
                                         nn.Sigmoid())
            
        
        elif self.arch_shape == 'dense_3D_CNN_ReLu_2nd_kernel_size':
            
            depth_kernel_size = self.get_n_th_smallest_denominator(2)
            
            num_layers = int((self.input_depth-1)/(depth_kernel_size-1))
            conv_layers = [nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(depth_kernel_size, 1, 1), stride=1, padding=0) for _ in range(num_layers)]
            self.encoder = nn.Sequential(*conv_layers,
                                         nn.Sigmoid())
            

        
        elif self.arch_shape == 'dense_3D_CNN_ReLu_kernel_size_5':
            depth_kernel_size = 5
            num_layers = int((self.input_depth-1)/(depth_kernel_size-1))
            last_kernel = int(self.input_depth - num_layers*(depth_kernel_size-1))
            conv_layers = [nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(depth_kernel_size, 1, 1), stride=1, padding=0) for _ in range(num_layers)]
            self.encoder = nn.Sequential(*conv_layers,
                                         nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(last_kernel, 1, 1), stride=1, padding=0),
                                         nn.Sigmoid())
            
        elif self.arch_shape == 'dense_3D_CNN_ReLu_kernel_size_8':
            depth_kernel_size = 8
            num_layers = int((self.input_depth-1)/(depth_kernel_size-1))
            last_kernel = int(self.input_depth - num_layers*(depth_kernel_size-1))
            conv_layers = [nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(depth_kernel_size, 1, 1), stride=1, padding=0) for _ in range(num_layers)]
            self.encoder = nn.Sequential(*conv_layers,
                                         nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(last_kernel, 1, 1), stride=1, padding=0),
                                         nn.Sigmoid())
            
                    
        elif self.arch_shape == 'dense_3D_CNN_ReLu_kernel_size_20':
            depth_kernel_size = 20
            num_layers = int((self.input_depth-1)/(depth_kernel_size-1))
            last_kernel = int(self.input_depth - num_layers*(depth_kernel_size-1))
            conv_layers = [nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(depth_kernel_size, 1, 1), stride=1, padding=0) for _ in range(num_layers)]
            self.encoder = nn.Sequential(*conv_layers,
                                         nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(last_kernel, 1, 1), stride=1, padding=0),
                                         nn.Sigmoid())
            
            
        elif self.arch_shape == 'dense_3D_CNN_ReLu':
            #depth_kernel_size = self.depth_kernel_size  ##TODO add to variable
            num_layers = int((self.input_depth-1)/(self.kernel_size-1))
            last_kernel = int(self.input_depth - num_layers*(self.kernel_size-1))
            
            while last_kernel <= 3:
                num_layers = num_layers - 1
                last_kernel = int(self.input_depth - num_layers*(self.kernel_size-1))
                
                
            conv_layers = [nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(self.kernel_size, 1, 1), stride=1, padding=0).to(self.model_dtype) for _ in range(num_layers)]
            self.encoder = nn.Sequential(*conv_layers)
            
            self.ecs_pred_model = nn.Sequential(nn.Linear(last_kernel,1).to(self.model_dtype),
                                                nn.Sigmoid())  ##* outputs of Linear is all negative
            
            self.ecs_classif_model = nn.Linear(last_kernel,2).to(self.model_dtype)  ###* nn.softmax(dim=-1) is applied inside crossentropyloss


    def get_n_th_smallest_denominator(self,n):
        j=0
        for i in range(2, self.input_depth//2 + 1):
            if (self.input_depth -1) % i == 0:
                depth_kernel_size = i + 1
                ##*cf output dim of conv3d
                j = j+1
            if j == n:
                return depth_kernel_size

        
if __name__ == '__main__':
    checkpoint_path = '/homes/o23gauvr/Documents/thèse/code/FASCINATION/outputs/accoustic_predictor/2024-04-04_15-22/checkpoints/val_loss=0.02-epoch=970.ckpt'
    input_depth = 107
    model = AcousticPredictor.load_from_checkpoint(checkpoint_path = checkpoint_path,input_depth = input_depth)
    
from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
import xarray as xr
import pandas as pd
import numpy as np
import hydra
import itertools
from src.utils import get_cfg_from_from_ckpt_path, check_differentiable
from src.CNN_2D import CNN_2D
from src.CNN_3D import CNN_3D
from src.CNN_with_classif_3D import CNN_with_classif_3D
import torch.nn.functional as F
import src.differentiable_fonc as DF
import src.loss.loss_func as LF

class AcousticPredictor(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model_hparams: dict, 
                 loss_name: str,
                 loss_hparams: dict,
                 opt_fn: dict,
                 ecs_classif_ckpt_path: str = None, 
                 loss_weight: dict = {"pred_weight": 1, "classif_weight": 1}, 
                 mask_type: str = "None"):
        
            
        super().__init__()
        self.model_dict = dict(Dense_2D_CNN_ReLu = Dense_CNN_2D, ECS_explicit_pred_3D = ECS_explicit_pred_3D, CNN_with_classif_3D = CNN_with_classif_3D, CNN_2D = CNN_2D, CNN_3D = CNN_3D) #Dense_CNN_with_classif_3D = Dense_CNN_with_classif_3D
        train_classif_dict = dict(Dense_2D_CNN_ReLu = False, ECS_explicit_pred_3D = True,  CNN_with_classif_3D = True, CNN_2D = False, CNN_3D = False) 
        
        self.loss_dict = dict(BCELoss = LF.BCELoss, DiceLoss = LF.DiceLoss, DiceBCELoss = LF.DiceBCELoss)
        self.opt_fn = opt_fn
        self.classif_weight = loss_weight["classif_weight"]
        self.pred_weight = loss_weight["pred_weight"]       
        self.mask_type = mask_type

        
        
        self.train_ecs_classification = train_classif_dict[model_name]
            
        if self.train_ecs_classification is False:
            cfg_ecs_classif = get_cfg_from_from_ckpt_path(ecs_classif_ckpt_path)
            self.ecs_classif_mod = hydra.utils.call(cfg_ecs_classif.model)
            self.ecs_classif_mod.load_state_dict(torch.load(ecs_classif_ckpt_path)['state_dict'])
            self.ecs_classif_mod = self.ecs_classif_mod.eval()
            for param in self.ecs_classif_mod.parameters():
                param.requires_grad = False  # Ensure no gradients are calculated for this model
        

            
        self.model = self.create_model(model_name, model_hparams)
        self.model_dtype = self.model.model_dtype ##* needed for trainer summary
        self.loss_func = self.initiate_loss(loss_name, loss_hparams)
        
        self.save_hyperparameters()
        
        
    def forward(self, ssf_input):
        # Forward function that is run when visualizing the graph
        return self.model(ssf_input)
    
    
    
    def configure_optimizers(self):
        return self.opt_fn(self)

    def cosanneal_lr_adamw(self, lr, T_max, weight_decay=0.):
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay= weight_decay)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=T_max,
            ),
        }


    def training_step(self, batch, batch_idx):
        return self.step(batch,'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch,'val')
    
    def test_step(self, batch, batch_idx):
        
        pred_loss, classif_loss = self.step(batch,'test')
        ecs_rmse = torch.sqrt(pred_loss)*670.25141631
        self.log("ECS RMSE", ecs_rmse, on_epoch = True)
        self.log("ECS classification corss entropy loss",classif_loss, on_epoch = True)
    
    
    def step(self, batch, phase = ""):
        ssp_input, ecs_truth = batch
        ecs_classif_truth = ecs_truth.detach().clone()
        ecs_classif_truth[ecs_classif_truth != 0] = 1
              
        
        if self.train_ecs_classification:
            ecs_pred, ecs_classif = self.model(ssp_input.requires_grad_(True))   
            
 
        else:
            ecs_pred = self.model(ssp_input)
            ecs_classif = self.ecs_classif_mod(ssp_input)  #(nn.ReLU()(ecs_pred - 0.5))
            
        if self.mask_type == "simple":
            ecs_classif = ecs_classif * ecs_pred
        
        if self.mask_type in ["None", None]:
            pass

            
              
        pred_loss = nn.MSELoss()(ecs_pred, ecs_truth)
        classif_loss = self.loss_func(ecs_classif, ecs_classif_truth)
        full_loss = self.pred_weight*pred_loss + self.classif_weight*classif_loss
        
        self.log(f"prediction loss {phase}", pred_loss,  prog_bar=False, on_step=None, on_epoch=True)
        self.log(f"classification loss {phase}", classif_loss,  prog_bar=False, on_step=None, on_epoch=True)
        self.log(f"{phase}_loss", full_loss,  prog_bar=True, on_step=None, on_epoch=True)
        
        if phase =="test":
            return pred_loss, classif_loss
    
        else:
            return full_loss

    
    def on_train_start(self):
        x,_ = next(iter(self.trainer.train_dataloader))
        check_differentiable(x, self.model, verbose = True)


    
    def create_model(self,model_name, model_hparams):
        if model_name in self.model_dict:
            return self.model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(self.model_dict.keys())}'
            
            
    def initiate_loss(self, loss_name, loss_hparams):
        if loss_name in self.loss_dict:
            return self.loss_dict[loss_name](**loss_hparams)
        else:
            assert False, f'Unknown loss name "{loss_name}". Available models are: {str(self.loss_dict.keys())}'
         
       
        
        
        
            
class Dense_CNN_2D(nn.Module):
    
    def __init__(self,
                    num_layers: int = 4,
                    input_depth: int = 107,
                    acoustic_variables: int = 1,
                    dtype_str: str = "float32"):
        
        super().__init__()
        
        self.model_dtype =  getattr(torch, dtype_str)
        in_ch = np.linspace(input_depth, acoustic_variables, num_layers + 1, dtype = np.int8)

        layers = []
        for i in range( num_layers ):
            layers.append(nn.Conv2d(in_channels= in_ch[i], out_channels= in_ch[i+1], kernel_size=1, stride=1, padding=0, dtype = self.model_dtype))
                          
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        
    
    def forward(self,x):
        return self.net(x)
    
    
    
class ECS_explicit_pred_3D(nn.Module):
    
    
    def __init__(self,
                 depth_array: np.array):
        super().__init__()
        self.model_dtype = getattr(torch, "float32")
        self.bias = torch.nn.Parameter(torch.empty(0))
        self.depth_array = depth_array
    
    
    def forward(self,ssp):
        ssp = ssp.unsqueeze(1)
        #print("After unsqueeze:", ssp.requires_grad, ssp.grad_fn)
        
        kernel = torch.tensor([-1.0, 1.0]).float().view(1,1,2,1,1).to(ssp.device)
        derivative = F.conv3d(ssp, kernel, padding=(0,0,0))
        #print("After first conv3d (derivative):", derivative.requires_grad, derivative.grad_fn)       
        
        sign = DF.differentiable_sign(derivative)
        #tau = 100
        #sign = torch.sign(derivative) + F.tanh(tau * derivative) - F.tanh(tau * derivative).detach()
        #print("After torch.sign (sign):", sign.requires_grad, sign.grad_fn)
        
        #sign_diff = sign[1:] - sign[:-1]
        sign_diff = F.conv3d(sign, kernel, padding=(1,0,0))
        #print("After third conv3d (sign_diff):", sign_diff.requires_grad, sign_diff.grad_fn)
        
        
        sign_change = F.tanh(10*F.relu(-sign_diff))
        #print("After torch.tanh (sign_change):", sign_change.requires_grad, sign_change.grad_fn)
        

        for pattern in ([1, 0, 1], [1, -1, 0, 0]):  ##* res_mak can also be used for a better complexity
            n = len(pattern)
            kernel_matrix = torch.eye(n)
            element_match = 0
            for i in range(n):
                kernel_element = kernel_matrix[i,:].view(1,1,n,1,1).to(ssp.device)
                element_match = element_match + (F.conv3d(sign, kernel_element, padding=(0,0,0)) - pattern[i])**2

            pattern_recognition = F.pad( element_match, (0, 0, 0, 0, 1, (sign_change.shape[2]- element_match.shape[2]) - 1),value=1.)    
            mask_discontinuity = 1 - F.relu(pattern_recognition+1) * F.relu(1-pattern_recognition)

            sign_change = sign_change * mask_discontinuity

        #print("After element-wise multiplication (sign_change):", sign_change.requires_grad, sign_change.grad_fn)
        

        mask = F.relu(2 - torch.cumsum(sign_change, dim=2))
        #print("After torch.cumsum (mask):", mask.requires_grad, mask.grad_fn)
        
        
        depth_array_tens = torch.tensor(np.expand_dims(self.depth_array[:mask.shape[2]], axis = (0,2,3))).to(ssp.device).type(sign_change.dtype)
        depth_array_tens[0,0,0,0] = 0.  ##TODO the true first z value is equal to 48cm. It may have to be considered that way
        ecs_pred = (sign_change * mask ).squeeze(dim=1)
        ecs_pred = (ecs_pred * depth_array_tens).max(dim=1).values / 670.25141631
        #print("After ecs_pred calculation:", ecs_pred.requires_grad, ecs_pred.grad_fn)
        
        ecs_classif = F.tanh(100*ecs_pred)
        #ecs_classif[ecs_classif != 0.] = 1.
        #print("After ecs_classif calculation:", ecs_classif.requires_grad, ecs_classif.grad_fn)
        
        
        return ecs_pred, ecs_classif
        


# def res_mak(inp,mat_t):
#     sum_pat = torch.sum(torch.abs(mat_t))
#     mat = mat_t.clone()
#     for i in range(mat.shape[-1]):
#         if mat[:,:,i]==0:
#             mat[:,:,i]=-(i**2.3+5)*10
#     x= f.conv1d(inp,mat)
#     x= x-sum_pat +1
#     x= torch.nn.ReLU()(x)
#     y= f.conv1d(inp,mat)
#     y= torch.nn.ReLU()(sum_pat+1-y)
#     z = x*y
#     return z
                
                
                
# class Dense_CNN_with_classif_3D(nn.Module):

#     def __init__(self,
#                 num_layers: int = 4,
#                 num_classes: int = 2,
#                 in_ch: int = 3,
#                 channels_start: int = 20,
#                 acoustic_variables: int = 1,
#                 dtype_str: str = "float32",
#                 loss_weight: dict = {"no_ecs_weight": 1, "ecs_weight": 100}
#                 ):
        
#         super().__init__()
#         self.train_ecs_classification = True
#         self.model_dtype =  getattr(torch, dtype_str)
#         self.in_ch = in_ch
#         self.no_ecs_weight = np.float32(loss_weight["no_ecs_weight"])
#         self.ecs_weight = np.float32(loss_weight["ecs_weight"])
        
#         layers = [nn.Conv3d(in_ch, channels_start, kernel_size=(3, 1, 1), padding=(1,0,0), dtype = self.model_dtype)]

#         channels = channels_start

#         for _ in range(num_layers -1):
#             layers.append(nn.Conv3d(in_channels = channels, out_channels = channels + 20, kernel_size=(3, 1, 1), padding=(1,0,0), dtype = self.model_dtype))
#             channels = channels + 20
            
            
#         layers.append(nn.Sequential(nn.Linear(channels, acoustic_variables).to(self.model_dtype),
#                                     nn.Sigmoid())  ##* outputs of Linear is all negative
#         )
        
#         layers.append(nn.Sequential(nn.Linear(channels, num_classes).to(self.model_dtype),
#                                     nn.Softmax(dim=1)) 
#         )
        
#         self.net = nn.Sequential(*layers)
        
        
#     def forward(self,x):
            
#         x = x.unsqueeze(1).repeat(1, self.in_ch, 1, 1, 1)
        
#         x = self.net[:-2](x)
        
#         ecs_pred = self.net[-2](x.permute(0,2,3,4,1))
#         ecs_classif = self.net[-1](x.permute(0,2,3,4,1))
        
#         ecs_pred, ecs_classif = ecs_pred.permute(0,4,1,2,3).squeeze(dim=1), ecs_classif.permute(0,4,1,2,3).squeeze(dim=1)
        
#         return ecs_pred, ecs_classif
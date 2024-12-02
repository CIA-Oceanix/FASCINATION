


import pytorch_lightning as pl

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import hydra 

from src.utils import get_cfg_from_from_ckpt_path
 


class LearnableGaussianKernel1d(pl.LightningModule):
    def __init__(self,
                 ae_ckpt_path: str,
                 opt_fn: dict,
                 train_on_ecs: bool,
                 depth_array = np.array,
                 kernel_size = 5, 
                 init_sigma=3.0,
                 flip_tensor: bool = True,
                 dtype_str = "float32"):
        
        super().__init__()
        #self.kernel_size_param = nn.Parameter(torch.tensor((kernel_size - 1) / 2, dtype=torch.float32), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([[init_sigma]], dtype=torch.float32), requires_grad=True)
        self.kernel_size = kernel_size
        self.model_dtype = getattr(torch, dtype_str)

        self.flip_tensor = flip_tensor
        
        self.norm_stats = None
        
        self.opt_fn = opt_fn
        self.train_on_ecs = train_on_ecs
        self.save_hyperparameters()


        
        self.ecs_explicit_model = ECS_explicit_pred_3D(depth_array)
        self.ecs_explicit_model = self.ecs_explicit_model.eval()
        for param in self.ecs_explicit_model.parameters():
            param.requires_grad = False  # Ensure no gradients are calculated for this model
    

        
        cfg_ae_model = get_cfg_from_from_ckpt_path(ae_ckpt_path)
        self.ae_model = hydra.utils.call(cfg_ae_model.model)
        self.ae_model.load_state_dict(torch.load(ae_ckpt_path)['state_dict'])
        self.ae_model = self.ae_model.eval()
        for param in self.ae_model.parameters():
            param.requires_grad = False  # Ensure no gradients are calculated for this model
    


        
    def forward(self, ssp_3D):
                       
 
        ssp_3D = ssp_3D.permute((0,2,3,1))
        ssp_3D_shape = ssp_3D.shape
        ssp_depth = ssp_3D_shape[-1]
        
        ssp_flatten = ssp_3D.reshape(-1,ssp_depth)
        
        if self.flip_tensor:
            mirrored_ssp = torch.flip(ssp_flatten, dims=[1])
            ssp_flatten = torch.cat((mirrored_ssp, ssp_flatten, mirrored_ssp), dim=1)            
        
        
        gausian_kernel = kornia.filters.get_gaussian_kernel1d(self.kernel_size, self.sigma) 
        pad = (self.kernel_size-1)//2
        ssp_filtered = F.conv1d(ssp_flatten.unsqueeze(1), weight= gausian_kernel.view(1,1,-1).to(ssp_flatten.device), padding=pad).squeeze(dim=1)
        
        if self.flip_tensor:
            ssp_filtered = ssp_filtered[:, ssp_depth:2*ssp_depth]
        
        ssp_3D_filtered = ssp_filtered.reshape(ssp_3D_shape).permute((0,3,1,2))
        
        
        return ssp_3D_filtered
        


    def configure_optimizers(self):
        return self.opt_fn(self)


    def cosanneal_lr_adamw(self, lr, T_max, weight_decay=0.):
        opt = torch.optim.AdamW([self.sigma], lr=lr, weight_decay= weight_decay)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=T_max,
            ),
        }



    def training_step(self, batch, batch_idx):
        filtered_loss = self.step(batch,'train')
        sigma_grad = torch.autograd.grad(outputs=filtered_loss, inputs=self.sigma,
                                         retain_graph=True, 
                                         create_graph=True,
                                         allow_unused=False)
        self.log("sigma gradient:", sigma_grad[0].item())
        return filtered_loss
    
    def on_train_epoch_end(self):
        # Log the current values of the parameters
        self.log('kernel_size', self.kernel_size)
        self.log('sigma', self.sigma.item())
            
    def validation_step(self, batch, batch_idx):
        return self.step(batch,'val')
    
    def test_step(self, batch): 
        
        ssp_truth, ssp_filtered = self.step(batch,'test')
        
        ssp_truth_unorm = self.unorm_ssp(ssp_truth)
        ssp_filtered_unorm = self.unorm_ssp(ssp_filtered)
        
        ecs_truth = self.ecs_explicit_model(ssp_truth_unorm)
        ecs_filtered = self.ecs_explicit_model(ssp_filtered_unorm)
        
        ssp_rmse = torch.sqrt(nn.MSELoss()(ssp_truth_unorm, ssp_filtered_unorm))
        
            
        self.log("SSP RMSE", ssp_rmse, on_epoch = True)
        
        ecs_rmse = torch.sqrt(nn.MSELoss()(ecs_truth, ecs_filtered))*670.25141631
        self.log("ECS RMSE", ecs_rmse, on_epoch = True)
        
        self.log('epoch_sigma', self.sigma.item())
        
        
    def on_test_start(self):
        self.norm_stats = self.trainer.datamodule.norm_stats
        

    
    def step(self, batch, phase = ""):
        
        ssp_truth, _ = batch 
    
        if phase == "train":
            ssp_truth = ssp_truth.requires_grad_(True)
        
        ssp_pred = self.ae_model(ssp_truth)
        ssp_filtered = self(ssp_pred)
        
        if self.train_on_ecs:
            
            ssp_truth_unorm = self.unorm_ssp(ssp_truth)
            ssp_filtered_unorm = self.unorm_ssp(ssp_filtered)
            
            ecs_truth = self.ecs_explicit_model(ssp_truth_unorm)
            ecs_filtered = self.ecs_explicit_model(ssp_filtered_unorm)
            
            filtered_loss = nn.MSELoss()(ecs_truth, ecs_filtered)
        
        else:
            filtered_loss = nn.MSELoss()(ssp_filtered, ssp_truth)
   
        self.log(f"{phase}_loss", filtered_loss, prog_bar=False, on_step=None, on_epoch=True)

        if phase == "test":
            return ssp_truth, ssp_filtered
        
        else:
            return filtered_loss
        


    def unorm_ssp(self, ssp):
        
        ssp_dtype = ssp.dtype
        
        if self.norm_stats is None:
            return ssp
        
        elif self.norm_stats["method"] == "min_max":
            x_min, x_max = self.norm_stats["params"]["x_min"], self.norm_stats["params"]["x_max"] 
            ssp = ssp *(x_max - x_min) + x_min
            
        elif self.norm_stats["method"] == "mean_std":
            mean, std = self.norm_stats["params"]["mean"], self.norm_stats["params"]["std"] 
            ssp = ssp*std + mean
            
        elif self.norm_stats["method"] == "mean_std_along_depth":
            mean = torch.tensor(self.norm_stats["params"]["mean"].data.reshape(1,-1,1,1), 
                                device = ssp.device)
            std = torch.tensor(self.norm_stats["params"]["std"].data.reshape(1,-1,1,1), 
                               device = ssp.device)
            ssp = ssp*std + mean
            
        return ssp.to(ssp_dtype)

class ECS_explicit_pred_3D(nn.Module):
    
    
    def __init__(self,
                 depth_array: np.array):
        super().__init__()
        self.model_dtype = getattr(torch, "float32")
        self.bias = torch.nn.Parameter(torch.empty(0))
        self.depth_array = depth_array
        self.tau = 100
    
    
    def forward(self,ssp):
        ssp = ssp.unsqueeze(1)        
        kernel = torch.tensor([-1.0, 1.0]).float().view(1,1,2,1,1).to(ssp.device)
        derivative = F.conv3d(ssp, kernel, padding=(0,0,0))
        
        #sign = DF.differentiable_sign(derivative)

        sign = torch.sign(derivative) + F.tanh(self.tau * derivative) - F.tanh(self.tau * derivative).detach()
        #print("After torch.sign (sign):", sign.requires_grad, sign.grad_fn)
        

        sign_diff = F.conv3d(sign, kernel, padding=(1,0,0))
        sign_change = F.tanh(10*F.relu(-sign_diff))

        for pattern in ([1, 0, 1], [1, -1, 0, 0]):  
            n = len(pattern)
            kernel_matrix = torch.eye(n)
            element_match = 0
            for i in range(n):
                kernel_element = kernel_matrix[i,:].view(1,1,n,1,1).to(ssp.device)
                element_match = element_match + (F.conv3d(sign, kernel_element, padding=(0,0,0)) - pattern[i])**2

            pattern_recognition = F.pad( element_match, (0, 0, 0, 0, 1, (sign_change.shape[2]- element_match.shape[2]) - 1),value=1.)    
            mask_discontinuity = 1 - F.relu(pattern_recognition+1) * F.relu(1-pattern_recognition)

            sign_change = sign_change * mask_discontinuity


        mask = F.relu(2 - torch.cumsum(sign_change, dim=2))

        depth_array_tens = torch.tensor(np.expand_dims(self.depth_array[:mask.shape[2]], axis = (0,2,3))).to(ssp.device).type(sign_change.dtype)
        depth_array_tens[0,0,0,0] = 0.  ##TODO the true first z value is equal to 48cm. It may have to be considered that way
        ecs_pred = (sign_change * mask ).squeeze(dim=1)
        ecs_pred = (ecs_pred * depth_array_tens).max(dim=1).values /670.25141631

        
        
        return ecs_pred
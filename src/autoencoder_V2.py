import pytorch_lightning as pl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from src.explicit_ecs import ECS_explicit_pred_1D, ECS_explicit_pred_3D
#from src.model.autoencoder.AE_CNN_2D import AE_CNN_2D
from src.model.autoencoder.AE_CNN_3D import AE_CNN_3D
#from src.model.autoencoder.AE_CNN_pool_2D import AE_CNN_pool_2D
#from src.model.autoencoder.AE_CNN_1D import AE_CNN_1D
from src.utils import check_differentiable, check_abnormal_grad
from torch.utils.tensorboard import SummaryWriter


import src.differentiable_fonc as DF

class AutoEncoder(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model_hparams: dict,
                 opt_fn :dict,
                 dim: str,
                 loss_weight: dict,
                 ):
    
        super().__init__()
        
        self.model_dict = dict( AE_CNN_3D = AE_CNN_3D) #AE_CNN_2D = AE_CNN_2D, AE_CNN_pool_2D  = AE_CNN_pool_2D, AE_CNN_1D = AE_CNN_1D #Dense_CNN_with_classif_3D = Dense_CNN_with_classif_3D
        self.dim = dim
        self.verbose = False

        self.loss_weight = loss_weight
        self.prediction_weight = nn.Parameter(torch.tensor(self.loss_weight['prediction_weight'], dtype=torch.float32,requires_grad=True))
        self.weighted_weight = nn.Parameter(torch.tensor(self.loss_weight['weighted_weight'], dtype=torch.float32, requires_grad=True))
        self.gradient_weight = nn.Parameter(torch.tensor(self.loss_weight['gradient_weight'], dtype=torch.float32, requires_grad=True))
        self.max_position_weight = nn.Parameter(torch.tensor(self.loss_weight['max_position_weight'], dtype=torch.float32, requires_grad=True))
        self.max_value_weight = nn.Parameter(torch.tensor(self.loss_weight['max_value_weight'], dtype=torch.float32, requires_grad=True))
        self.inflection_pos_weight = nn.Parameter(torch.tensor(self.loss_weight['inflection_pos_weight'], dtype=torch.float32, requires_grad=True))
        self.inflection_value_weight = nn.Parameter(torch.tensor(self.loss_weight['inflection_value_weight'], dtype=torch.float32, requires_grad=True))
        self.fft_weight = nn.Parameter(torch.tensor(self.loss_weight['fft_weight'], dtype=torch.float32, requires_grad=True))


        self.opt_fn = opt_fn
        
        self.depth_pre_treatment = {"method": None}


        self.model_AE = self.initiate_model(model_name, model_hparams)
        self.encoder, self.decoder = self.model_AE.encoder, self.model_AE.decoder
        self.model_dtype = self.model_AE.model_dtype ##* needed for trainer summary
        
        self.save_hyperparameters()
        

        
        
    def forward(self, x):

            
        x_hat = self.model_AE(x)

        return x_hat
    


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

        batch.requires_grad = True
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx,'val')
    
    def test_step(self, batch, batch_idx):

        self.step(batch, batch_idx,'test')
    
        ssp_truth = batch
        with torch.no_grad():
            ssp_pred = self(ssp_truth)
        
        
        ssp_rmse = torch.sqrt(torch.mean((self.unorm(ssp_pred)-self.unorm(ssp_truth))**2))            
        self.log("SSP RMSE", ssp_rmse, on_epoch = True)

        
        self.log("prediction test mse",  nn.MSELoss()(ssp_pred, ssp_truth), on_epoch = True)
        
        if len(self.z_tens) > 1:
            coordinates = (self.z_tens,)
            ssp_gradient_truth = torch.gradient(input = ssp_truth, spacing = coordinates, dim=1)[0]
            ssp_gradient_pred = torch.gradient(input = ssp_pred, spacing = coordinates, dim=1)[0]
            self.log("gradient test mse", nn.MSELoss()(ssp_gradient_truth, ssp_gradient_pred) , on_epoch = True)  
            
        
        # ecs_truth = self.ecs_explicit_model(ssp_truth)
        # ecs_pred =  self.ecs_explicit_model(ssp_pred)
        # ecs_loss =  nn.MSELoss()(ecs_pred, ecs_truth)            
        # self.log(f"ecs test mse", ecs_loss, on_epoch=True)


        original_max_value, original_max_pos = torch.max(self.ssp_truth, dim=1)
        reconstructed_max_value, reconstructed_max_pos = torch.max(ssp_pred, dim=1)
        max_position_loss =  nn.MSELoss()(original_max_pos.float(), reconstructed_max_pos.float())            
        self.log(f"max position test mse", max_position_loss, on_epoch=True)

        max_value_loss =  nn.MSELoss()(original_max_value, reconstructed_max_value) 
        self.log(f"max value test loss", max_value_loss, on_epoch=True)

        ecs_rmse = torch.sqrt(torch.mean((self.z_tens[original_max_pos]-self.z_tens[reconstructed_max_pos])**2))
        self.log("ECS RMSE", ecs_rmse, on_epoch = True)


        ssp_truth_fft = torch.fft.fft(ssp_truth, dim=1)
        ssp_pred_fft = torch.fft.fft(ssp_pred, dim=1)
        fft_loss = torch.mean((torch.abs(ssp_truth_fft) - torch.abs(ssp_pred_fft)) ** 2)
        #fft_loss =  nn.MSELoss()(torch.abs(ssp_truth_fft), torch.abs(ssp_pred_fft)) 
        self.log(f"fft test mse", fft_loss, on_epoch=True)


        signal_length = ssp_truth.shape[1]
        weights = torch.ones(signal_length, device=ssp_truth.device, dtype=ssp_truth.dtype)
        decay_factor = 0.1 
        max_significant_depth = 300
        max_significant_depth_idx = torch.searchsorted(self.z_tens, max_significant_depth, right=False)
        weights[:max_significant_depth_idx] = 1.0  # Strong emphasis on the first max_significant_depth_idx points
        weights[max_significant_depth_idx:] = torch.exp(-decay_factor * torch.arange(max_significant_depth_idx, signal_length))
        weights = weights.view(1, 1, -1, 1, 1)
        
        weighted_loss = torch.mean(weights * (ssp_truth - ssp_pred) ** 2)
        self.log(f"weighted mse loss test mse", weighted_loss, on_epoch=True)


        inflection_points_truth_mask = DF.differentiable_min_max_search(ssp_truth,dim=1,tau=10)
        inflection_points_pred_mask = DF.differentiable_min_max_search(ssp_pred,dim=1,tau=10)
        index_tensor = torch.arange(0, signal_length, device=ssp_truth.device, dtype=ssp_truth.dtype).view(1, -1, 1, 1)
        truth_inflex_pos = (inflection_points_truth_mask * index_tensor).sum(dim=1)/inflection_points_truth_mask.sum(dim=1)
        pred_inflex_pos = (inflection_points_pred_mask * index_tensor).sum(dim=1)/inflection_points_pred_mask.sum(dim=1)
        inflection_pos_loss = nn.MSELoss()(pred_inflex_pos, truth_inflex_pos)
        self.log(f"inflection position test mse", inflection_pos_loss, on_epoch=True)


        inflection_value_loss = nn.MSELoss(reduction="none")(ssp_pred,ssp_truth)*inflection_points_truth_mask
        inflection_value_loss = inflection_value_loss.mean()
        self.log(f"inflection value test mse", inflection_value_loss, on_epoch=True)




    def full_loss(self, ssp_truth, ssp_pred):
        # Apply softplus to weights to ensure positivity
        prediction_weight = F.softplus(self.prediction_weight)
        weighted_weight = F.softplus(self.weighted_weight)
        gradient_weight = F.softplus(self.gradient_weight)
        max_position_weight = F.softplus(self.max_position_weight)
        max_value_weight = F.softplus(self.max_value_weight)
        inflection_pos_weight = F.softplus(self.inflection_pos_weight)
        inflection_value_weight = F.softplus(self.inflection_value_weight)
        fft_weight = F.softplus(self.fft_weight)

        # Compute prediction loss
        pred_loss = F.mse_loss(ssp_pred, ssp_truth)
        self.log("prediction_loss", pred_loss, prog_bar=False, on_step=None, on_epoch=True)
        full_loss = prediction_weight * pred_loss

        # Compute max position loss
        original_max_value, original_max_pos = torch.max(ssp_truth, dim=1)
        reconstructed_max_value, reconstructed_max_pos = torch.max(ssp_pred, dim=1)
        max_position_loss = F.mse_loss(original_max_pos.float(), reconstructed_max_pos.float()) 
        self.log("max_position_loss", max_position_loss, prog_bar=False, on_step=None, on_epoch=True)
        if max_position_weight != 0:
            full_loss += max_position_weight * max_position_loss

        # Compute max value loss
        max_value_loss = F.mse_loss(original_max_value, reconstructed_max_value) 
        self.log("max_value_loss", max_value_loss, prog_bar=False, on_step=None, on_epoch=True)
        if max_value_weight != 0:
            full_loss += max_value_weight * max_value_loss

        # Compute weighted loss
        weights = torch.ones(ssp_truth.shape[1], device=ssp_truth.device, dtype=ssp_truth.dtype)
        decay_factor = 0.1 
        max_significant_depth_idx = torch.searchsorted(self.z_tens, self.max_significant_depth, right=False)
        weights[:max_significant_depth_idx] = 1.0  # Strong emphasis on the first max_significant_depth_idx points
        weights[max_significant_depth_idx:] = torch.exp(-decay_factor * torch.arange(
            max_significant_depth_idx, ssp_truth.shape[1],
            device=ssp_truth.device, dtype=ssp_truth.dtype))
        weights = weights.view(1, 1, -1, 1, 1)
        
        weighted_loss = torch.mean(weights * (ssp_truth - ssp_pred) ** 2)
        self.log("weighted_mse_loss", weighted_loss, prog_bar=False, on_step=None, on_epoch=True)
        if weighted_weight != 0:
            full_loss += weighted_weight * weighted_loss

        # Compute gradient loss if applicable
        if hasattr(self, 'z_tens') and len(self.z_tens) > 1:
            coordinates = (self.z_tens,)
            ssp_gradient_truth = torch.gradient(input=ssp_truth, spacing=coordinates, dim=1)[0]
            ssp_gradient_pred = torch.gradient(input=ssp_pred, spacing=coordinates, dim=1)[0]
            grad_loss = F.mse_loss(ssp_gradient_truth, ssp_gradient_pred)
            self.log("gradient_test_mse", grad_loss, on_epoch=True)
            if gradient_weight != 0:
                full_loss += gradient_weight * grad_loss

        # Compute ECS loss
        if hasattr(self, 'ecs_explicit_model') and self.loss_weight.get('ecs_weight', 0) != 0:
            ecs_truth = self.ecs_explicit_model(ssp_truth)
            ecs_pred = self.ecs_explicit_model(ssp_pred)
            ecs_loss = F.mse_loss(ecs_pred, ecs_truth)            
            self.log("ecs_loss", ecs_loss, prog_bar=False, on_step=None, on_epoch=True)         
            full_loss += fft_weight * ecs_loss  # Replace with ecs_weight if applicable

        # Compute FFT loss
        ssp_truth_fft = torch.fft.fft(ssp_truth, dim=1)
        ssp_pred_fft = torch.fft.fft(ssp_pred, dim=1)
        fft_loss = torch.mean((torch.abs(ssp_truth_fft) - torch.abs(ssp_pred_fft)) ** 2)
        self.log("fft_loss", fft_loss, prog_bar=False, on_step=None, on_epoch=True)
        if fft_weight != 0:            
            full_loss += fft_weight * fft_loss

        # Compute inflection position loss
        inflection_points_truth_mask = DF.differentiable_min_max_search(ssp_truth, dim=1, tau=10)
        inflection_points_pred_mask = DF.differentiable_min_max_search(ssp_pred, dim=1, tau=10)
        signal_length = ssp_truth.shape[1]
        index_tensor = torch.arange(0, signal_length, device=ssp_truth.device, dtype=ssp_truth.dtype).view(1, -1, 1, 1)
        truth_inflex_pos = (inflection_points_truth_mask * index_tensor).sum(dim=1) / inflection_points_truth_mask.sum(dim=1)
        pred_inflex_pos = (inflection_points_pred_mask * index_tensor).sum(dim=1) / inflection_points_pred_mask.sum(dim=1)
        inflection_pos_loss = F.mse_loss(pred_inflex_pos, truth_inflex_pos)
        self.log("inflection_pos_loss", inflection_pos_loss, prog_bar=False, on_step=None, on_epoch=True)

        if inflection_pos_weight != 0:
            full_loss += inflection_pos_weight * inflection_pos_loss

        # Compute inflection value loss
        inflection_value_loss = F.mse_loss(ssp_pred, ssp_truth, reduction="none") * inflection_points_truth_mask
        inflection_value_loss = inflection_value_loss.mean()
        self.log("inflection_value_loss", inflection_value_loss, prog_bar=False, on_step=None, on_epoch=True)

        if inflection_value_weight != 0:
            full_loss += inflection_value_weight * inflection_value_loss

        return full_loss



    def step(self, batch, batch_idx, phase='train'):
        self.ssp_truth = batch
        ssp_pred = self(self.ssp_truth)



        # Logging additional weights for monitoring
        self.log("prediction_weight", F.softplus(self.prediction_weight), on_step=True, on_epoch=True)
        self.log("weighted_weight", F.softplus(self.weighted_weight), on_step=True, on_epoch=True)
        self.log("gradient_weight", F.softplus(self.gradient_weight), on_step=True, on_epoch=True)
        self.log("max_position_weight", F.softplus(self.max_position_weight), on_step=True, on_epoch=True)
        self.log("max_value_weight", F.softplus(self.max_value_weight), on_step=True, on_epoch=True)
        self.log("inflection_pos_weight", F.softplus(self.inflection_pos_weight), on_step=True, on_epoch=True)
        self.log("inflection_value_weight", F.softplus(self.inflection_value_weight), on_step=True, on_epoch=True)
        self.log("fft_weight", F.softplus(self.fft_weight), on_step=True, on_epoch=True)

        # Compute full loss
        full_loss = self.full_loss(self.ssp_truth, ssp_pred)

        # Logging the total loss
        self.log(f"{phase}_loss", full_loss, prog_bar=True, on_step=True, on_epoch=True)

        return full_loss


    def setup(self, stage=None):

        batch = next(iter(self.trainer.datamodule.train_dataloader())).to(self.device)
        self.example_input_array = batch
        self.depth_pre_treatment = self.trainer.datamodule.depth_pre_treatment
        self.norm_stats = self.trainer.datamodule.norm_stats
        self.depth_arr = self.trainer.datamodule.depth_array
        self.z_tens = torch.tensor(self.trainer.datamodule.coords["z"].data, device=batch.device,dtype=batch.dtype)

        self.max_significant_depth = 300

        if self.depth_pre_treatment["method"] == "pca":
            tens_shape = torch.Size([batch.shape[0], len(self.depth_arr), *batch.shape[2:]])
            pca = self.depth_pre_treatment["fitted_pca"]
            self.dif_pca_4D = DF.Differentiable4dPCA(pca, original_shape=tens_shape, device=batch.device,dtype=batch.dtype)     
            
            self.max_significant_depth = 10 ###TODO a faire varier, mettre en hyperparametres

        # if self.loss_weight['ecs_weight'] != 0:
        #     if self.dim == "1D":
        #         self.ecs_explicit_model = ECS_explicit_pred_1D(self.depth_array)
        #     else:
        #         self.ecs_explicit_model = ECS_explicit_pred_3D(self.depth_array)


        
    def on_train_start(self):
        batch = next(iter(self.trainer.datamodule.train_dataloader())).to(self.device)
        check_differentiable(batch,self, self.full_loss, verbose=False,raise_error=True)


        # with torch.no_grad():
        #     conv_weights = self.encoder.net[0].weight
        #     conv_weights.fill_(0)
        #     conv_weights[:,:, :, 3, 3] = 1  

        #     self.encoder.net[0].bias[:] = 0

        #     conv_weights = self.decoder.net[0].weight
        #     conv_weights.fill_(0)
        #     conv_weights[:,:, 0, 3, 3] = 1  

        #     self.decoder.net[0].bias[:] = 0




    def on_after_backward(self):   
        #writter = SummaryWriter(log_dir=f"{self.trainer.logger.log_dir}/backward_grads")  
        writter = self.trainer.logger.experiment 
        check_abnormal_grad(model = self, input = self.ssp_truth, writter = writter, verbose=True,raise_error=False)
        
    
    
    def initiate_model(self,model_name, model_hparams):
        if model_name in self.model_dict:
            return self.model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(self.model_dict.keys())}'

    
    def unorm(self, ssp_tens):


        
        if self.norm_stats["method"] == "min_max":
            x_min, x_max = self.norm_stats["params"]["x_min"], self.norm_stats["params"]["x_max"] 
            ssp_tens = ssp_tens*(x_max - x_min) + x_min

            
        elif self.norm_stats["method"] == "mean_std":
            mean, std = self.norm_stats["params"]["mean"], self.norm_stats["params"]["std"] 
            ssp_tens = ssp_tens*std + mean

        elif self.norm_stats["method"] == "mean_std_along_depth":
            mean, std = torch.tensor(self.norm_stats["params"]["mean"].reshape(1,-1,1,1), device = ssp_tens.device, dtype=ssp_tens.dtype),torch.tensor(self.norm_stats["params"]["std"].reshape(1,-1,1,1), device = ssp_tens.device, dtype=ssp_tens.dtype)
            ssp_tens = ssp_tens*std + mean


        if self.depth_pre_treatment["method"] == "pca":

            self.dif_pca_4D.original_shape = torch.Size([ssp_tens.shape[0], len(self.depth_arr), *ssp_tens.shape[2:]])
            ssp_tens = self.dif_pca_4D.inverse_transform(ssp_tens)
    
    
        return ssp_tens



import pytorch_lightning as pl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from src.explicit_ecs import ECS_explicit_pred_1D, ECS_explicit_pred_3D
#from src.model.autoencoder.AE_CNN_2D import AE_CNN_2D
from src.model.autoencoder.AE_CNN_3D import AE_CNN_3D
from src.model.autoencoder.AE_CNN import AE_CNN

#from src.model.autoencoder.AE_CNN_pool_2D import AE_CNN_pool_2D
#from src.model.autoencoder.AE_CNN_1D import AE_CNN_1D
from src.utils import check_differentiable, check_abnormal_grad


import src.differentiable_fonc as DF
from src.loss.loss_func import *
from src.activation_function import ScaledTanh


class AutoEncoder(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model_hparams: dict,
                 opt_fn :dict,
                 loss_weight: dict
                 ):
    
        super().__init__()
        
        self.model_dict = dict(AE_CNN = AE_CNN, AE_CNN_3D = AE_CNN_3D) #AE_CNN_2D = AE_CNN_2D, AE_CNN_pool_2D  = AE_CNN_pool_2D, AE_CNN_1D = AE_CNN_1D #Dense_CNN_with_classif_3D = Dense_CNN_with_classif_3D
        self.verbose = False

        self.loss_weight = loss_weight
        self.opt_fn = opt_fn
        self.model_name = model_name
        self.model_hparams = model_hparams
        self.model_AE = self.initiate_model(self.model_name, self.model_hparams)
        self.model_dtype = self.model_AE.model_dtype 


        
        self.depth_pre_treatment = {"method": None}



        
        self.save_hyperparameters()

        self.dif_pca_4D = DF.Differentiable4dPCA()     

        


    def setup(self, stage=None):

        batch = next(iter(self.trainer.datamodule.train_dataloader())).to(self.device)
        self.example_input_array = batch

        self.depth_pre_treatment = self.trainer.datamodule.depth_pre_treatment
        self.norm_stats = self.trainer.datamodule.norm_stats
        self.depth_arr = self.trainer.datamodule.depth_array
        self.z_tens = torch.tensor(self.depth_arr, device=batch.device,dtype=batch.dtype)

        
        if stage == 'fit':
            self.model_hparams['input_shape'] = batch.shape
            self.model_hparams['device'] = batch.device 
            self.model_AE = self.initiate_model(self.model_name, self.model_hparams)
            self.set_last_activation_fucntion()
            self.encoder, self.decoder = self.model_AE.encoder, self.model_AE.decoder

            check_differentiable(batch, self, verbose=False, raise_error=True)




        self.max_significant_depth = 300

        if self.depth_pre_treatment["method"] == "pca":
            tens_shape = torch.Size([batch.shape[0], len(self.depth_arr), *batch.shape[2:]])
            pca = self.depth_pre_treatment["fitted_pca"]
            self.dif_pca_4D = DF.Differentiable4dPCA(pca, batch_shape=tens_shape ,device=batch.device,dtype=batch.dtype)     
            
            if self.depth_pre_treatment["train_on"] == "components":
                variance_to_explain = 0.95
                cumsum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                self.max_significant_depth = np.argmax(cumsum_explained_variance_ratio >= variance_to_explain) + 1

                self.z_tens = torch.tensor(range(1, pca.n_components + 1), device=batch.device,dtype=batch.dtype) 


        # if self.loss_weight['ecs_weight'] != 0:
        #     if self.dim == "1D":
        #         self.ecs_explicit_model = ECS_explicit_pred_1D(self.depth_array)
        #     else:
        #         self.ecs_explicit_model = ECS_explicit_pred_3D(self.depth_array)
        
        
    def forward(self, x):

        if self.depth_pre_treatment["method"] == "pca":

            x = self.dif_pca_4D.transform(x)

            if self.verbose:
                n_components = self.dif_pca_4D.n_components
                print(f"PCA pre treatment, depth components: {n_components}")
            
        x_hat = self.model_AE(x)


        if self.depth_pre_treatment["method"] == "pca": 
            x_hat = self.dif_pca_4D.inverse_transform(x_hat)
            

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


    def on_train_start(self):
        pass
        # self.model_AE = self.initiate_model(self.model_name, self.model_hparams)
        # self.set_last_activation_fucntion()
        # self.encoder, self.decoder = self.model_AE.encoder, self.model_AE.decoder

        # with torch.no_grad():
        #     # Initialize encoder weights and bias for identity function
        #     conv_weights = self.encoder.net[0].weight
        #     conv_weights.fill_(0)
        #     conv_weights[:, :, 3, 3, 3] = 1  # Assuming kernel size is 7x7x7

        #     self.encoder.net[0].bias.fill_(0)

        #     # Initialize decoder weights and bias for identity function
        #     conv_weights = self.decoder.net[0].weight
        #     conv_weights.fill_(0)
        #     conv_weights[:, :, 3, 3, 3] = 1  # Assuming kernel size is 7x7x7

        #     self.decoder.net[0].bias.fill_(0)


        

    def on_test_start(self):
        pass

            

    def training_step(self, batch, batch_idx):
        self.train()
        batch.requires_grad = True
        return self.step(batch,'train')
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        return self.step(batch,'val')
    
    def test_step(self, batch, batch_idx):
        self.eval()
        return self.step(batch,'test')
    


    
    def step(self, batch, phase = ""):
        
        self.ssp_truth = batch   
        ssp_reconstructed = self(self.ssp_truth)

        if self.depth_pre_treatment["method"] == "pca" and self.depth_pre_treatment["train_on"] == "components":
            self.ssp_truth = self.dif_pca_4D.transform(self.ssp_truth)
            ssp_reconstructed = self.dif_pca_4D.transform(ssp_reconstructed)


        pred_loss = nn.MSELoss()(ssp_reconstructed, self.ssp_truth)
        self.log(f"prediction loss", pred_loss, prog_bar=False, on_step=None, on_epoch=True)

        weighted_loss = weighted_mse_loss(ssp_reconstructed, self.ssp_truth, self.z_tens, self.max_significant_depth)
        self.log(f"weighted loss", weighted_loss, prog_bar=False, on_step=None, on_epoch=True)


        max_value_loss, max_position_loss = max_position_and_value_loss(self.ssp_truth, ssp_reconstructed)
        self.log(f"max position loss", max_position_loss, prog_bar=False, on_step=None, on_epoch=True)
        self.log(f"max value loss", max_value_loss, prog_bar=False, on_step=None, on_epoch=True)
                    
        full_loss = self.loss_weight['prediction_weight'] * pred_loss 
        + self.loss_weight['weighted_weight'] * weighted_loss 
        + self.loss_weight['max_position_weight'] * max_position_loss 
        + self.loss_weight['max_value_weight'] * max_value_loss

        if self.depth_pre_treatment["method"] == "pca" and self.depth_pre_treatment["train_on"] == "components":

            pass

        else:

            gradient_loss = gradient_mse_loss(self.ssp_truth, ssp_reconstructed, self.z_tens)
            self.log(f"gradient loss", gradient_loss, prog_bar=False, on_step=None, on_epoch=True)

            min_max_pos_loss, min_max_value_loss = min_max_position_and_value_loss(self.ssp_truth, ssp_reconstructed)
            self.log(f"min max position loss", min_max_pos_loss, prog_bar=False, on_step=None, on_epoch=True)
            self.log(f"min max value loss", min_max_value_loss, prog_bar=False, on_step=None, on_epoch=True)

            fft_loss = fourier_loss(ssp_reconstructed, self.ssp_truth)
            self.log(f"fft loss", fft_loss, prog_bar=False, on_step=None, on_epoch=True)

            full_loss = full_loss + self.loss_weight['gradient_weight'] * gradient_loss 
            + self.loss_weight['min_max_position_weight'] * min_max_pos_loss 
            + self.loss_weight['min_max_value_weight'] * min_max_value_loss 
            + self.loss_weight['fft_weight'] * fft_loss

        
        if phase == "test":
        
            if self.depth_pre_treatment["method"] == "pca" and self.depth_pre_treatment["train_on"] == "components":
                self.ssp_truth = self.dif_pca_4D.inverse_transform(self.ssp_truth)
                ssp_reconstructed = self.dif_pca_4D.inverse_transform(ssp_reconstructed)

            unorm_ssp_reconstructed = self.unorm(ssp_reconstructed)
            unorm_ssp_truth = self.unorm(self.ssp_truth)

            ssp_rmse = torch.sqrt(torch.mean((unorm_ssp_reconstructed-unorm_ssp_truth)**2))            
            self.log("SSP RMSE", ssp_rmse, on_epoch = True, reduce_fx = torch.mean)

            truth_max_pos = torch.argmax(unorm_ssp_truth, dim=1).detach().cpu().numpy()
            reconstructed_max_pos = torch.argmax(unorm_ssp_reconstructed, dim=1).detach().cpu().numpy()
            ecs_rmse = np.sqrt(np.mean((self.depth_arr[truth_max_pos]-self.depth_arr[reconstructed_max_pos])**2))
            self.log("ECS RMSE", ecs_rmse, on_epoch = True, reduce_fx = torch.mean)


        self.log(f"{phase}_loss", full_loss, prog_bar=False, on_step=None, on_epoch=True)
        

        return full_loss
        



    def on_after_backward(self):   
        #writter = SummaryWriter(log_dir=f"{self.trainer.logger.log_dir}/backward_grads")  
        writter = self.trainer.logger.experiment 
        check_abnormal_grad(model = self, input = self.ssp_truth, writter = writter, verbose=True,raise_error=False)
        
    

    def initiate_model(self,model_name, model_hparams):
        if model_name in self.model_dict:
            return self.model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(self.model_dict.keys())}'


    def set_last_activation_fucntion(self):

        if self.norm_stats["method"] == "min_max":
            self.model_AE.decoder.net[-1] = nn.Sigmoid()
        
        elif self.norm_stats["method"] == "mean_std" or self.norm_stats["method"] == "mean_std_along_depth":

            data_min = self.trainer.datamodule.min_val
            data_max = self.trainer.datamodule.max_val
            scale = max(abs(data_min), abs(data_max))
            scale = scale + 0.1*scale
            self.model_AE.decoder.net[-1] = ScaledTanh(scale) #nn.Identity() #ScaledTanh(scale)  ##See with nn.Identity()

        if self.model_AE.use_final_act_fn == False:
            self.model_AE.decoder.net[-1] = nn.Identity()

         



    def unorm(self, ssp_tens):


        if self.depth_pre_treatment["norm_on"] == "components":
            ssp_tens = self.dif_pca_4D.transform(ssp_tens)

        if self.norm_stats["method"] == "min_max":
            x_min, x_max = self.norm_stats["params"]["x_min"], self.norm_stats["params"]["x_max"] 
            ssp_tens = ssp_tens*(x_max - x_min) + x_min
            
        elif self.norm_stats["method"] == "mean_std":
            mean, std = self.norm_stats["params"]["mean"], self.norm_stats["params"]["std"] 
            ssp_tens = ssp_tens*std + mean

        elif self.norm_stats["method"] == "mean_std_along_depth":
            mean, std = torch.tensor(self.norm_stats["params"]["mean"].reshape(1,-1,1,1), device = ssp_tens.device, dtype=ssp_tens.dtype),torch.tensor(self.norm_stats["params"]["std"].reshape(1,-1,1,1), device = ssp_tens.device, dtype=ssp_tens.dtype)
            ssp_tens = ssp_tens*std + mean

        if self.depth_pre_treatment["norm_on"] == "components": 
            ssp_tens = self.dif_pca_4D.inverse_transform(ssp_tens)

    
        return ssp_tens
    

    
# def test_non_deterministic(func, input_tensor, num_tests=10):
#     # Set random seeds for reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)

#     # Run the function multiple times and store the outputs
#     outputs = []
#     for _ in range(num_tests):
#         output = func(input_tensor)
#         outputs.append(output)

#     # Compare the outputs
#     for i in range(1, num_tests):
#         if not torch.equal(outputs[0], outputs[i]):
#             print("The function is non-deterministic.")
#             return

#     print("The function is deterministic.")



# def list_model_functions(model):
#     print("Modules (Layers) in the model:")
#     list_modules = []
#     for name, module in model.named_modules():
#         #print(f"{name}: {module}")
#         list_modules.append((name,module))
    
#     return list_modules

#     print("\nParameters in the model:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.shape}")

# def check_model_deterministic(model, input_tensor):

#     list_modules = list_model_functions(model)
#     for name, module in list_modules:
#         test_non_deterministic(module, input_tensor)


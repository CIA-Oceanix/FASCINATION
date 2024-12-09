from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.UNet_3D import UNet_3D
from src.UNet_2D import UNet_2D
from src.CNN_2D import CNN_2D
from src.CNN_3D import CNN_3D
import src.loss.loss_func as LF
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



class ECS_classification(pl.LightningModule):
    def __init__(self, model_name, model_hparams, opt_fn: dict, depth_array: np.array, loss_name: str, loss_hparams: dict):
        
        """
        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        
        super().__init__()
        
        self.model_dict = dict(UNet_3D = UNet_3D, UNet_2D = UNet_2D, CNN_2D = CNN_2D, CNN_3D = CNN_3D)
        self.loss_dict = dict(BCELoss = LF.BCELoss, DiceLoss = LF.DiceLoss, DiceBCELoss = LF.DiceBCELoss)
        self.opt_fn = opt_fn
        self.depth_array = depth_array
        #self.loss_weight = loss_weight

        
        #self.test_batches = dict(ssf_input_list = [], ecs_classif_list = [])
        
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.create_model(model_name, model_hparams)
        self.loss_func = self.initiate_loss(loss_name, loss_hparams)
               
        self.model_dtype = self.model.model_dtype ##* needed for trainer summary

    

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
        
        ssf_input, ecs_classif_truth = batch
        ecs_classif = self.model(ssf_input)
        
        ecs_preds = self.get_predictions(ecs_classif)

        ecs_preds = ecs_preds.flatten().cpu().numpy()
        ecs_target = ecs_classif_truth.flatten().cpu().numpy()
        
        precision = precision_score(ecs_target, ecs_preds)
        recall = recall_score(ecs_target, ecs_preds)
        f1 = f1_score(ecs_target, ecs_preds)
        dice = 2 * (precision * recall) / (precision + recall)
        
        log_dict = {'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'dice_coefficient': dice
                    }
        
        self.log_dict(log_dict, on_step=None, on_epoch=True)
        

    
    def step(self, batch, phase = ""):    
        ssf_input, ecs_classif_truth = batch
        ecs_classif = self.model(ssf_input)
        #classif_loss = nn.CrossEntropyLoss(weight = torch.tensor([self.no_ecs_weight, self.ecs_weight]).to(ecs_classif.device))(ecs_classif, ecs_classif_truth) ##* a softmax is applied at the end of the network in addition of the logsoftmax inside hte loss function
        classif_loss = self.loss_func(ecs_classif, ecs_classif_truth)
        self.log(f"{phase}_loss", classif_loss,  prog_bar=True, on_step=None, on_epoch=True)
        
        return classif_loss



    # def on_test_epoch_end(self):
    #     pass
        
        
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
            
            
    def get_predictions(self, outputs, threshold=0.5):
        """
        Convert model outputs to binary predictions based on the given threshold.
        """
        return (outputs > threshold).float()
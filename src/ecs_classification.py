from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.UNet import UNet



class ECS_classification(pl.LightningModule):
    def __init__(self, model_name, model_hparams, opt_fn: dict):
        """

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        
        self.model_dict = dict(UNet = UNet)
        
        self.opt_fn = opt_fn
        
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.create_model(model_name, model_hparams)
        
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
        return self.step(batch,'test')
    
    
    def step(self, batch, phase = ""):    
        ssf_input, ecs_classif_truth = batch
        ecs_classif = self.model(ssf_input)
        classif_loss = nn.NLLLoss()(ecs_classif, ecs_classif_truth) ##* a softmax is applied at the end of the network
        
        return classif_loss
        
        
    def create_model(self,model_name, model_hparams):
        if model_name in self.model_dict:
            return self.model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(self.model_dict.keys())}'
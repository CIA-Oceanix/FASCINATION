import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(pl.LightningModule):

    def __init__(self, opt_fn, arch_shape = '4_15', final_act_func = 'sigmoid', norm_stats=None):
        super().__init__()

        
        #self.solver = solver
        #self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        #self.metrics = test_metrics or {}
        #self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        
        self.arch_shape = arch_shape
        self.final_act_func = final_act_func
        
        
        
        # self.acoustic_predictor = acoustic_predictor
        # if self.acoustic_predictor != None:
        #     self.acoustic_predictor.eval()

        self.architecture(arch_shape)

        
        # if torch.cuda.is_available():
        #     self.input_da.to('cuda')
        # self.log_model_summary()
        


    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule._norm_stats
            ### TODO: check if this gets the rights parameters for each split
        return (0., 1.)


    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]
    
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        m, s = self.norm_stats()
        batch=batch.squeeze()
        out = self(batch)
        loss = torch.sqrt(F.mse_loss(out*s+m,batch*s+m))
        # self.test_data.append(torch.stack(
        #     [
        #         batch.cpu()*s+m,
        #         out.cpu()*s+m
        #     ],
        #     dim=1
        # ))
        
        self.log('test_rmse', loss, on_step= False, on_epoch=True)
        
    
    
    def forward(self, batch):
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded
    
    

    def step(self, batch, phase =""):
        # if self.training and batch.isfinite().float().mean() < 0.9:
        #     return None, None
        ##TODO: manage this test
        
        loss, out = self.ae_step(batch.squeeze(), phase)  ##! for conv2D the patch should be of size 1 and we sueeze the tensor

        training_loss = loss
        return training_loss, out
        

    def ae_step(self, batch, phase=""):
        out = self(batch=batch)
        #loss = self.weighted_mse(out - batch.tgt, self.rec_weight)n n 
        loss = F.mse_loss(out,batch)

        with torch.no_grad():
            #self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_mse", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss, out


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
                nn.Conv2d(8, 4, kernel_size=3, padding=1),
                nn.ReLU() 
            )
            
            ###TODO: enlever ReLU
            ###TODO check stride
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 107, kernel_size=2, stride=2),
                nn.Sigmoid() ###! sortie [0;1] ? test softplus, htgt, relu
            )
            
                      
        
        
        if arch_shape == "4_15_test": 
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
    # def log_model_summary(self):
    #     self.log(summary(self,input_size = (107,240,240)))
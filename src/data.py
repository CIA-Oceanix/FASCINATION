import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
from collections import namedtuple

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class AutoEncoderDatamodule(pl.LightningDataModule):
    def __init__(self, input_da, domains, dl_kw):
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.dl_kw = dl_kw

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False

    def setup(self, stage):
        if not self.is_data_normed:
            train_data = self.input_da.isel(self.domains['train'])
            for var in self.input_da.data_vars:
                mean, std = self.norm_stats(train_data[var])
                self.input_da = (self.input_da - mean)/std
            self.is_data_normed = True

        if stage == "fit":
            self.train_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['train'])
            )
            self.val_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['val'])
            )
        if stage == "test":
            self.val_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['val'])
            )
            self.test_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['test'])
            )


    def norm_stats(self, da):
        mean = da.mean()
        std = da.std()
        return mean, std
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    
class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, da):
        super().__init__()
        self.da = da

    def __len__(self):
        return len(self.da.time)
    
    def __getitem__(self, index):
        return TrainingItem._make((np.nan_to_num(self.da.celerity[index].data.astype(np.float32)), np.nan_to_num(self.da.celerity[index].data.astype(np.float32))))
    
class AcousticPredictorDatamodule(pl.LightningDataModule):
    def __init__(self, input_da, dl_kw):
        super().__init__()
        self.input = input_da[0]
        self.target = input_da[1]
        self.dl_kw = dl_kw

        self.test_time = None
        self.test_var = None
        self.test_lat = None
        self.test_lon = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False
    
    def setup(self, stage):
        random_dataset = AcousticPredictorDataset(self.input, self.target)
        train_da, val_da, test_da = torch.utils.data.random_split(random_dataset, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))
        if not self.is_data_normed:
            input_train, target_train = self.input.isel(time=train_da.indices), self.target.isel(time=train_da.indices)
            #mean, std = self.norm_stats(input_train, target_train)
            self.input = (self.input - 472.33156028)/(1552.54994512 - 472.33156028) # hard coded values for now because it saves computation time
            self.target["cutoff_freq"] = (self.target["cutoff_freq"])/10000  
            self.target["ecs"] = (self.target["ecs"])/670.25141631
            self.is_data_normed = True
        
        if stage == 'fit':
            self.train_ds = AcousticPredictorDataset(
                self.input.isel(time=train_da.indices), self.target.isel(time=train_da.indices)
                )
            self.val_ds = AcousticPredictorDataset(
                self.input.isel(time=val_da.indices), self.target.isel(time=val_da.indices)
            )
        if stage == 'test':
            self.val_ds = AcousticPredictorDataset(
                self.input.isel(time=val_da.indices), self.target.isel(time=val_da.indices)
            )
            self.test_ds = AcousticPredictorDataset(
                self.input.isel(time=test_da.indices), self.target.isel(time=test_da.indices)
            )
            self.test_time = self.test_ds.variables["time"]
            self.test_var = self.test_ds.variables["variable"]
            self.test_lat = self.test_ds.variables["lat"]
            self.test_lon = self.test_ds.variables["lon"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    
    def norm_stats(self, input, target):
        mean, std = {}, {}
        mean["input"] = input.mean()
        std["input"] = input.std()
        for j in target.data_vars:
            mean[j] = target[j].mean()
            std[j] = target[j].std()

        return mean, std

class AcousticPredictorDataset(torch.utils.data.Dataset):
    def __init__(self, volume, variables):
        super().__init__()
        self.volume, self.variables = volume.transpose('time', 'z', 'lat', 'lon'), variables.to_array().transpose('time', 'variable', 'lat', 'lon')

    def __len__(self):
        return min(len(self.volume.time), len(self.variables.time))
    
    def __getitem__(self, index):
        return TrainingItem._make((np.nan_to_num(self.volume.celerity[index].data.astype(np.float32)), self.variables[index].data.astype(np.float32)))

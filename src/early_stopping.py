from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer import Trainer


class ValidEarlyStopping(EarlyStopping):

    def __init__(self, monitor='val_loss', patience=50, mode='min', verbose=False, min_epochs=100):
        super().__init__(monitor=monitor, patience=patience, mode=mode, verbose=verbose)
        self.min_epochs = min_epochs
    
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.min_epochs:
            super().on_validation_end(trainer, pl_module)

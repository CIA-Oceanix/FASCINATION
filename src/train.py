import torch
import pickle
import os
from torchinfo import summary
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, test_dm=None, ckpt=None, pickle_path = None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    # if torch.cuda.is_available():
    #     lit_mod.to('cuda')
        
    #batch_size = dm.dl_kw.batch_size
    model_summary = summary(lit_mod,
                            input_size = (365,107,240,240), 
                            device = lit_mod.device.type, 
                            batch_dim = None, 
                            col_names = ["input_size","output_size","num_params","params_percent","mult_adds"], 
                            verbose = 1)

    #trainer.logger.experiment.log_hparams(dict(summary = str(model_summary)))
    
    
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_dm is None:
        test_dm = dm

    # best_ckpt_path = trainer.checkpoint_callback.best_model_path
    # trainer.callbacks = []
    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')
    
    with open(f"{trainer.logger.log_dir}/model_summary.log", 'w+') as f:
        f.write(str(model_summary))

    if pickle_path:
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path,"wb") as f:
            pickle.dump(
                dict(
                    train=dm.train_ds.volume.time.values,
                    val=dm.val_ds.volume.time.values,
                    test=dm.test_ds.volume.time.values
                ),
                f
            )
                
    
import torch
import pickle
import os
from torchinfo import summary
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, dim = "3D", test_dm=None, ckpt=None, pickle_path = None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    
    
    model_summary = summary(lit_mod,
                            input_size = lit_mod.model_AE.input_shape, 
                            device = lit_mod.device.type, 
                            batch_dim = None, 
                            dtypes=[lit_mod.model_dtype],
                            col_names = ["input_size","output_size","num_params","params_percent","mult_adds"], 
                            verbose = 1)

    
    with open(f"{trainer.logger.log_dir}/model_summary.log", 'w+') as f:
        f.write(str(model_summary))
    
    
    
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    # if test_dm is None:
    #     test_dm = dm


    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')
    


    if pickle_path:
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path,"wb") as f:
            pickle.dump(
                dict(
                    train=dm.train_ds.input.time.values,
                    val=dm.val_ds.input.time.values,
                    test=dm.test_ds.input.time.values
                ),
                f
            )
                
    
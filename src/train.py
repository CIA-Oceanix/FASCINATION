import torch
import pickle
import os
from torchinfo import summary
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, dim = "3D", test_dm=None, ckpt=None, save_dm = False):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    
    
    # model_summary = summary(lit_mod,
    #                         input_size = lit_mod.model_AE.input_shape, 
    #                         device = lit_mod.device.type, 
    #                         batch_dim = None, 
    #                         dtypes=[lit_mod.model_dtype],
    #                         col_names = ["input_size","output_size","num_params","params_percent","mult_adds"], 
    #                         verbose = 1)

    
    # with open(f"{trainer.logger.log_dir}/model_summary.log", 'w+') as f:
    #     f.write(str(model_summary))
    
    
    
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    # if test_dm is None:
    #     test_dm = dm


    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')
    


    if save_dm:
        dm_path = f"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_dm_157_141_240_good_split.pkl"#f"/Odyssey/private/o23gauvr/code/FASCINATION/pickle/enatl_natl_dm__{chn}_196_256.pkl" #enatl_dm_4_157_196_256.pkl
        os.makedirs(os.path.dirname(dm_path), exist_ok=True)
        with open(dm_path,"wb") as f:
            pickle.dump(dm,f
            )
                
    
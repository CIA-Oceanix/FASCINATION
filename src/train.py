import torch
from torchinfo import summary
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, test_dm=None, ckpt=None):
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

    trainer.logger.experiment.log_hparams(dict(summary = str(model_summary)))
    
    
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_dm is None:
        test_dm = dm

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.callbacks = []
    trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)
    
    # with open(f"{trainer.logger.log_dir}my_module_log.log", 'w+') as f:
    #     report = summary(lit_mod,input_size = (107,240,240), device = lit_mod.device.type)
    #     f.write(str(report))

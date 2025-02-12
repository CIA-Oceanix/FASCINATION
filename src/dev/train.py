import torch
from torchinfo import summary
torch.set_float32_matmul_precision('high')

def base_training(trainer, train_dm, test_dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    # if torch.cuda.is_available():
    #     lit_mod.to('cuda')
    
    model_summary = summary(lit_mod,
                            input_size = (test_dm.input_da.shape[-1],*test_dm.input_da.shape[:-1]), 
                            device = lit_mod.device.type, 
                            batch_dim = None, 
                            col_names = ["input_size","output_size","num_params","params_percent","mult_adds"], 
                            verbose = 1)

    #trainer.logger.experiment.log_hparams(dict(summary = str(model_summary)))
    
    
    trainer.fit(lit_mod, datamodule=train_dm, ckpt_path=ckpt)

    # if test_dm is None:
    #     test_dm = dm

    # best_ckpt_path = trainer.checkpoint_callback.best_model_path
    # trainer.callbacks = []
    trainer.test(lit_mod, datamodule=test_dm, ckpt_path='best')
    
    with open(f"{trainer.logger.log_dir}/model_summary.log", 'w+') as f:
        f.write(str(model_summary))
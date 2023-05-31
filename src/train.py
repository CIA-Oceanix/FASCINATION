import torch
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, test_dm=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_dm is None:
        test_dm = dm

    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.callbacks = []
    trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)
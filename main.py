import hydra
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import os

@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    
    
    config_file = f"{cfg.trainer.logger.save_dir}{cfg.trainer.logger.name}/{cfg.trainer.logger.version}/config.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    writer = SummaryWriter()
    writer.add_text("Hydra Config", OmegaConf.to_yaml(cfg))
    writer.close()

    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()


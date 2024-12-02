import hydra
import torch
from omegaconf import OmegaConf
import os

@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    

    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()


import hydra

@hydra.main(config_path='config', config_name='main', version_base='1.2')
def main(cfg):
    
    if cfg.model_architecture == "MLICPlusPlus":
        cfg.datamodule.reshape.method = "factor_64"
        
    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()
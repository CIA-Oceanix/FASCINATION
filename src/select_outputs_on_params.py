from pathlib import Path
from tqdm import tqdm
import shutil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_cfg_from_ckpt_path



def select_outputs_on_params(outputs_path, param_dict, dir_to_ignore):
    
    matched_models = []
    
    ckpt_list = list(Path(outputs_path).rglob('*.ckpt'))

    for ckpt_path in tqdm(ckpt_list):
        
        ckpt_path = str(ckpt_path)
    
        if any(dir_name in ckpt_path for dir_name in dir_to_ignore):
            print(f"Skipping {ckpt_path} because it contains a directory to ignore.")
            continue
    
        try:
            cfg = get_cfg_from_ckpt_path(ckpt_path, pprint = False)
        
        except:
            continue
            

        config_paths = {
            "model_name": cfg.model,
            "channels_list": cfg.model.model_hparams,
            "prediction_weight": cfg.model.loss_weight,
            "gradient_weight": cfg.model.loss_weight,
            "n_conv_per_layer": cfg.model.model_hparams,
            "padding": cfg.model.model_hparams,
            "interp_size": cfg.model.model_hparams,
            "pooling": cfg.model.model_hparams,
            "pooling_dim": cfg.model.model_hparams,
            "upsample_mode": cfg.model.model_hparams,
            "final_upsample_str": cfg.model.model_hparams,
            "act_fn_str": cfg.model.model_hparams,
            "final_act_fn_str": cfg.model.model_hparams,
            "linear_layer": cfg.model.model_hparams,
            "latent_size": cfg.model.model_hparams,
            "lr": cfg.model.opt_fn,
            "normalization_method": cfg.datamodule.norm_stats,
            "manage_nan": cfg.datamodule,
            "n_profiles": cfg.datamodule
        }


        mismatch_found = False
        
        
        for param in param_dict.keys():
            
            if param_dict[param] is not None:
                
                try:
                    
                    if param == "normalization_method":
                        cfg_param = config_paths[param].get("method")
                        
                    else:
                        cfg_param = config_paths[param].get(param)
                    

                    
                    if isinstance(param_dict[param], list):
                        if isinstance(cfg_param, list):
                            # Direct list-to-list comparison
                            if cfg_param != param_dict[param]:
                                mismatch_found = True
                                break
                            
                        else:
                            # Use "not in" when param_dict[param] is a list and cfg_param is a single item
                            if cfg_param not in param_dict[param]:
                                mismatch_found = True
                                break
                    
                    else:
                        # Use "!=" for comparison if param_dict[param] is not a list
                        if cfg_param != param_dict[param]:
                            mismatch_found = True
                            break
                                
                except:
                    mismatch_found = True
                    break
            
        if not mismatch_found:
            matched_models.append(ckpt_path)
            
        
    return matched_models
            
            
            

def copy_directory_contents(ckpt_list, destination_path):
    
    # Extract the base path up to the specific date directory
    
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)
    
    for ckpt_path in tqdm(ckpt_list):
        
        channels_list = get_cfg_from_ckpt_path(ckpt_path, pprint = False).model.model_hparams.channels_list
        model_destination_path = os.path.join(destination_path, f"channels_{channels_list}")
        
        base_dir = os.path.dirname(os.path.normpath(ckpt_path.split("checkpoints")[0]))
        # Copy all contents from the base directory to the destination path
        try:
            for item in os.listdir(base_dir):
                src_path = os.path.join(base_dir, item)
                dest_path = os.path.join(model_destination_path, item)
                
                if os.path.isdir(src_path):
                    # Copy directory and its contents
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                else:
                    # Copy individual files
                    shutil.copy2(src_path, dest_path)
            
            #print(f"All contents copied from {base_dir} to {destination_path}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
            
        


if __name__ == "__main__":
    
    output_path = "outputs/AE/AE_CNN_3D/"

    dir_to_ignore = [] #["[1,1]_trilinear_test_on_padding", "upsample_mode_test_on_[1,1]", "[1,1,2,2,4]_trilinear_test_on_padding"]
    
    copy_to_dir = True
        
    # param_dict = {"model_name": "AE_CNN_3D",
    #               "channels_list": [[1,1], [1,8]],
    #               "prediction_weight": 1,
    #               "gradient_weight": 0,
    #               "n_conv_per_layer": 1,
    #               "padding": "linear",
    #               "interp_size": None,
    #               "pooling": ["Max", "None"],
    #               "pooling_dim": "all",
    #               "final_upsample_str": "upsample_pooling",
    #               "act_fn_str": "None",
    #               "final_act_fn_str": "None",
    #               "lr": 0.001,
    #               "normalization_method": "min_max",
    #               "manage_nan": "suppress",
    #               "n_profiles": None,  
    #               }

    param_dict = {"model_name": "AE_CNN_3D",
                  "channels_list": [[1,8], [1,8,8], [1,8,8,8,8]],
                  "pooling": ["Max", "None"],
                  "pooling_dim": "all",
                  "manage_nan": "suppress",
                  }

    
    dest_dir =  f"outputs/AE/AE_CNN_3D/8_channels/"
    
    print("Searching matching models")
    matched_models = select_outputs_on_params(output_path, param_dict, dir_to_ignore)
    
    print(f"Founded matching models: {matched_models}")

    if copy_to_dir:
        
        print(f"Copying founded models to {dest_dir}")
        copy_directory_contents(matched_models, dest_dir)
        
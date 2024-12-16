import sys
import os

running_path = "/homes/o23gauvr/Documents/thèse/code/FASCINATION/"
sys.path.insert(0,running_path)
os.chdir(running_path)

import shutil
from src.utils import get_cfg_from_ckpt_path
from pathlib import Path
from tqdm import tqdm

def check_params(config, param_dict):
    for key, value in param_dict.items():
        if key not in config:
            return False
        if isinstance(value, dict):
            if not check_params(config[key], value):
                return False
        elif isinstance(value, list):
            if config[key] not in value:
                return False
        else:
            if config[key] != value:
                return False
    return True


def select_outputs_on_params(output_path, param_dict, dir_to_ignore = []):

    matched_models = []
    
    ckpt_list = list(Path(output_path).rglob('*.ckpt'))

    for ckpt_path in tqdm(ckpt_list):
        
        ckpt_path = str(ckpt_path)
    
        if any(dir_name in ckpt_path for dir_name in dir_to_ignore):
            print(f"Skipping {ckpt_path} because it contains a directory to ignore.")
            continue
    
        try:
            cfg = get_cfg_from_ckpt_path(ckpt_path, pprint = False)
        
        except:
            continue


        if check_params(cfg, param_dict):
            matched_models.append(ckpt_path)

    return matched_models



def copy_directory_contents(ckpt_list, destination_path):
    
    
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


def main():
    dir_to_ignore = []  # ["[1,1]_trilinear_test_on_padding", "upsample_mode_test_on_[1,1]", "[1,1,2,2,4]_trilinear_test_on_padding"]

    param_dict = {
        "model": {"loss_weight": {"prediction_weight": 0.5}},
        "datamodule": {
            "norm_stats": {"method": "mean_std_along_depth"},
            "depth_pre_treatment": {"params": 1, "norm_on": "components", "train_on": "components", "method": "pca"},
        },
    }

    outputs_path = "outputs/remote/AE_CNN_3D/pca-pre-treatment/pca_pre_treatment_50_epochs"

    copy_to_dir = True
    dest_dir = "outputs/AE/AE_CNN_3D/visualisation/pca_1_norm_on_components_train_on_components_mean_std_along_depth_loss_weight_05"

    print("Searching matching models")




    matched_models = select_outputs_on_params(outputs_path, param_dict, dir_to_ignore)
    
    print(f"Founded matching models: {matched_models}")

    if copy_to_dir:
        
        print(f"Copying founded models to {dest_dir}")
        copy_directory_contents(matched_models, dest_dir)

if __name__ == "__main__":
    main()
import os 
from pathlib import Path
from tqdm import tqdm
from utils import get_cfg_from_ckpt_path
import shutil

def safe_get(cfg, keys, default=None):
    """Tries to get a value from nested dictionary using a list of keys, returns default (None) if not found."""
    try:
        value = cfg
        for key in keys:
            value = value[key]
        return value
    except KeyError:
        return default
    

def relocate_outputs(outputs_path, dir_to_ignore,
                     save_dir =  "outputs/AE/AE_CNN_3D",
                    delete_src = False):
    
    ckpt_list = list(Path(outputs_path).rglob('*.ckpt'))

    for ckpt_path in tqdm(ckpt_list):
        
        ckpt_path = str(ckpt_path)
    
        if any(dir_name in ckpt_path for dir_name in dir_to_ignore):
            print(f"Skipping {ckpt_path} because it contains a directory to ignore.")
            continue
    
        try:
            cfg = get_cfg_from_ckpt_path(ckpt_path, pprint = False)
        
        except:
            #Path.rmdir(ckpt_path)
            continue
        
        # output_path = cfg["trainer"]["logger"]["save_dir"].split("AE")[0]
        # model_name = cfg['model']['model_name']

        channels_list = str(safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'channels_list']))
        prediction_weight = safe_get(cfg, ['model', 'loss_weight', 'prediction_weight'])
        gradient_weight = safe_get(cfg, ['model', 'loss_weight', 'gradient_weight'])
        max_weight = safe_get(cfg, ['model', 'loss_weight', 'max_position_weight'])

        n_conv_per_layer = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'n_conv_per_layer'])
        padding = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'padding'])
        interp_size = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'interp_size'])
        pooling = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'pooling'])

        pooling_dim = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'pooling_dim'], "all")

        depth_pre_treatment = safe_get(
            cfg, 
            ['datamodule', 'depth_pre_treatment', 'method'], None
        )
        
        if depth_pre_treatment:
            depth_pre_treatment = f"{depth_pre_treatment}_" \
                                f"n_components_{safe_get(cfg, ['datamodule', 'depth_pre_treatment', 'params'])}"

        fft_weight = safe_get(cfg, ['model', 'loss_weight', 'fft_weight'], 0)
        weighted_weight = safe_get(cfg, ['model', 'loss_weight', 'weighted_weight'], 0)
        inflexion_weight = safe_get(cfg, ['model', 'loss_weight', 'inflexion_weight'], 0)

        final_upsample_str = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'final_upsample_str'])
        upsample_mode = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'upsample_mode'])
        act_fn_str = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'act_fn_str'])
        final_act_fn_str = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'final_act_fn_str'])
        linear_layer = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'linear_layer'])
        latent_size = safe_get(cfg, ['model_config', 'model_hparams', 'AE_CNN_3D', 'latent_size'])
        lr = safe_get(cfg, ['model', 'opt_fn', 'lr'])
        normalization_method = safe_get(cfg, ['datamodule', 'norm_stats', 'method'])
        manage_nan = safe_get(cfg, ['datamodule', 'manage_nan'])
        n_profiles = safe_get(cfg, ['datamodule', 'n_profiles'])


        date = str(ckpt_path).split("/")[-3]



        dir_to_relocate = "/".join(str(ckpt_path).split("/")[:-2])


        # dest_dir =  f"{output_path}AE/" \
        #             f"{model_name}/" \
        dest_dir = f"{save_dir}/" \
                    f"channels_{channels_list}/" \
                    f"pred_{prediction_weight}_grad_{gradient_weight}_max_pos_{max_weight}_fft_{fft_weight}_weighted_{weighted_weight}_inflexion_{inflexion_weight}/" \
                    f"depth_pre_treatment_{depth_pre_treatment}/" \
                    f"upsample_mode_{upsample_mode}/" \
                    f"linear_layer_{linear_layer}_lattent_size_{latent_size}/" \
                    f"{n_conv_per_layer}_conv_per_layer/padding_{padding}/interp_size_{interp_size}/pooling_{pooling}_on_dim_{pooling_dim}/" \
                    f"final_upsample_{final_upsample_str}/act_fn_{act_fn_str}_final_act_fn_{final_act_fn_str}/" \
                    f"lr_{lr}/normalization_{normalization_method}/manage_nan_{manage_nan}/n_profiles_{n_profiles}/{date}"
                    # 
                    
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for item in os.listdir(dir_to_relocate):
            src = os.path.join(dir_to_relocate, item)
            dest = os.path.join(dest_dir, item)

            if src == dest:
                continue
            
            if delete_src:
                if os.path.isdir(dest) and os.path.exists(dest):
                    shutil.rmtree(dest)

                shutil.move(src, dest)
                
            else:
                if os.path.isdir(src):
                    shutil.copytree(src, dest, dirs_exist_ok=True)  # Copies directories and contents
                else:
                    shutil.copy2(src, dest)  # Copies individual files

                    
    if delete_src:
        print("Removing empty direcotries")
        remove_empty_dirs(outputs_path)          
        
            
            
def remove_empty_dirs(output_path):
    # Convert output_path to Path object
    path = Path(output_path)
    
    dir_list = list(sorted(path.rglob('*'), key=lambda p: len(p.parts), reverse=True))
    
    # Iterate over all subdirectories recursively
    for subdir in tqdm(dir_list):
        # Check if the path is a directory and if it's empty
        if subdir.is_dir() and not any(subdir.iterdir()):
            #print(f"Removing empty directory: {subdir}")
            os.rmdir(subdir)  # Remove the empty directory
            


if __name__ == "__main__":
    
    output_path = "outputs/AE/AE_CNN_3D/"
    save_dir = "/DATASET/envs/o23gauvr/outputs/AE/AE_CNN_3D/"

    dir_to_ignore = ["test"]
    
    print("Relocating outputs directories")
    relocate_outputs(output_path, dir_to_ignore, save_dir = save_dir ,delete_src = True)
    

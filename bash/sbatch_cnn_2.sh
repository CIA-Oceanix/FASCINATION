#!/bin/bash -l
#SBATCH --partition=Odyssey            
#SBATCH --job-name=UwU  
#SBATCH --gres=gpu:l40s:1     
#SBATCH --mem=60G     
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            
#SBATCH --exclusive=user


echo "Job started."


source /Odyssey/private/o23gauvr/start_conda.sh
conda activate run_model
echo "Environment activated successfully."


declare -a xp="autoencoder_V2"
declare -a model="AE_CNN"

declare -a features_list=([5000,3000,1000,5] [5000,3000,1000,10] [5000,3000,1000,100] [5000,3000,1000,157]) #[5000,2500,2] # [1,2,4,8]
declare -a loss_weights_list=("1,1,0,0,0,0,0,0,0,0")


declare -a norm_stats="mean_std_along_depth" 
declare -a lr=1e-3
declare -a dense=True
declare -a kernel_list=1
declare -a profile_ratio=null
declare -a pooling="None"
declare -a n_conv_per_layer=1
declare -a manage_nan="supress_with_max_depth"

declare -a current_branch=$(git rev-parse --abbrev-ref HEAD)
declare -a cuda=0
declare -a max_epoch=1000
declare -a root_dir="/Odyssey/private/o23gauvr/outputs"
declare -a save_dir="'$current_branch/$model'" 


# Added loss weights array (comma-separated values: pw,ww,gw,etw,mpw,mwv,mmmp,mmwv,fw,ecw)

for lw in "${loss_weights_list[@]}"
do
    IFS=',' read -r pw ww gw etw mpw mwv mmmp mmwv fw ecw <<< "$lw"
    for features in "${features_list[@]}"
    do
        HYDRA_FULL_ERROR=1 \
        srun python /Odyssey/private/o23gauvr/code/FASCINATION/main.py  \
        xp=$xp \
        root_save_dir=$root_dir \
        save_dir_name=$save_dir \
        trainer.max_epochs=$max_epoch \
        model.opt_fn.lr=$lr \
        datamodule.norm_stats.method=$norm_stats \
        datamodule.manage_nan=$manage_nan \
        model.loss_weight.prediction_weight=$pw \
        model.loss_weight.weighted_weight=$ww \
        model.loss_weight.gradient_weight=$gw \
        model.loss_weight.error_treshold_weight=$etw \
        model.loss_weight.max_position_weight=$mpw \
        model.loss_weight.max_value_weight=$mwv \
        model.loss_weight.min_max_position_weight=$mmmp \
        model.loss_weight.min_max_value_weight=$mmwv \
        model.loss_weight.fft_weight=$fw \
        model.loss_weight.ecs_weight=$ecw \
        model_config.model_hparams.$model.channels_list=$features \
        model_config.model_hparams.$model.kernel_list=$kernel_list \
        model_config.model_hparams.$model.dense=$dense \
        model_config.model_hparams.$model.pooling=$pooling \
        model_config.model_hparams.$model.n_conv_per_layer=$n_conv_per_layer \
        hydra.job.env_set.CUDA_VISIBLE_DEVICES=$cuda 
    done
done

echo "Job finished."



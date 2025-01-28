#!/bin/bash -l
#SBATCH --partition=Odyssey            
#SBATCH --job-name=UwU  
#SBATCH --gres=gpu:l40s:1       
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            
#SBATCH --exclusive=user


echo "Job started."


source /Odyssey/private/o23gauvr/start_conda.sh
conda activate run_model
echo "Environment activated successfully."





declare -a learning_rate_list=(0.0001 0.00001) # 0.00001 0.000001
declare -a channels_list=([107,100,50,20]) #[107,100,50]  [107,100] [107,100,50,20,10] # [1,2,4,8]

declare -a pooling_dim="depth"
declare -a norm_stats="mean_std"
declare -a dense=True
declare -a interp_size=0
declare -a linear_layer=False
declare -a cr=10000

declare -a current_branch=$(git rev-parse --abbrev-ref HEAD)
declare -a cuda=0
declare -a max_epoch=150
declare -a root_dir="/Odyssey/private/o23gauvr/outputs"
declare -a save_dir="'$current_branch/dense_on_depth_test_lr_on_pooled'" 
declare -a model="AE_CNN"


# if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
#     kinit -l 5d
#     krenew -K 10 &
# fi


for lr in "${learning_rate_list[@]}"
do

    for channels in "${channels_list[@]}"
    do


        HYDRA_FULL_ERROR=1 \
        srun python /Odyssey/private/o23gauvr/code/FASCINATION/main.py \
        root_save_dir=$root_dir \
        save_dir_name=$save_dir \
        trainer.max_epochs=$max_epoch \
        model.opt_fn.lr=$lr \
        model_config.model_hparams.$model.channels_list=$channels \
        model_config.model_hparams.$model.padding.interp_size=$interp_size \
        model_config.model_hparams.$model.linear_layer.use=$linear_layer \
        model_config.model_hparams.$model.linear_layer.cr=$cr \
        model_config.model_hparams.$model.dense=$dense \
        hydra.job.env_set.CUDA_VISIBLE_DEVICES=$cuda 
    done
done

echo "Job finished."



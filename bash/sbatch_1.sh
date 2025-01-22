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





declare -a learning_rate_list=(0.00001 0.000001) # 0.00001 0.000001


declare -a pooling_dim="depth"
declare -a norm_stats="mean_std"
declare -a interp_size=0

declare -a current_branch=$(git rev-parse --abbrev-ref HEAD)
declare -a cuda=0
declare -a max_epoch=100
declare -a root_dir="/Odyssey/private/o23gauvr/outputs"
declare -a save_dir="'$current_branch/test_on_depth'" 
declare -a model="AE_CNN"


# if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
#     kinit -l 5d
#     krenew -K 10 &
# fi


for lr in "${learning_rate_list[@]}"
do

        IFS=',' read -r pred_weight weighted_weight grad_weight max_position_weight max_value_weight min_max_position_weight min_max_value_weight fft_weight <<< "$weights"

        echo "(pred, weighted pred, grad, max position, max value, inflection value, inflection pos, fft) weights: $weights"

        HYDRA_FULL_ERROR=1
        srun python /Odyssey/private/o23gauvr/code/FASCINATION/main.py \
        root_save_dir=$root_dir \
        save_dir_name=$save_dir \
        trainer.max_epochs=$max_epoch \
        model.opt_fn.lr=$lr \
        model.loss_weight.prediction_weight=$pred_weight \
        model.loss_weight.weighted_weight=$weighted_weight \
        model.loss_weight.gradient_weight=$grad_weight \
        model.loss_weight.max_position_weight=$max_position_weight \
        model.loss_weight.max_value_weight=$max_value_weight \
        model.loss_weight.min_max_position_weight=$min_max_position_weight \
        model.loss_weight.min_max_value_weight=$min_max_value_weight \
        model.loss_weight.fft_weight=$fft_weight \
        hydra.job.env_set.CUDA_VISIBLE_DEVICES=$cuda \

done

echo "Job finished."



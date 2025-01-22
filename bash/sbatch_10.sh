#!/bin/bash
#SBATCH --partition=Odyssey            
#SBATCH --job-name=UwU  
#SBATCH --gres=gpu:a100:1       
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            
#SBATCH --exclusive=user



echo "Job started."


source /Odyssey/private/o23gauvr/start_conda.sh
source activate run_model
echo "Environment activated successfully."




declare -a cr_list=(1000) # 10000 1000 #  #[1,8] 
declare -a channels_list=([1,2,4,8] [1,2,4] [1,2,4,8,8]) # [1,2,4,8]
declare -a n_conv_per_layer=(1 2 3) # 2
declare -a kernel_list=([7,5,3])
declare -a loss_weights_list=("1,0,100,0,0,100,100,0")
declare -a learning_rate_list=(0.0001 0.00001 0.000001)
declare -a dropout_list=(0 0.5)
declare -a pooling=("conv" "Max")
#declare -a use_linear_layer=("True" "False")
declare -a use_final_act_func=("True" "False")


declare -a pooling_dim="all"
declare -a norm_stats="mean_std"
declare -a interp_size=5

declare -a current_branch=$(git rev-parse --abbrev-ref HEAD)
declare -a cuda=0
declare -a max_epoch=200
declare -a root_dir="/Odyssey/private/o23gauvr/outputs"
declare -a save_dir="'$current_branch/test_on_linear_200_epochs'" 
declare -a model="AE_CNN"


# if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
#     kinit -l 5d
#     krenew -K 10 &
# fi


for weights in "${loss_weights_list[@]}"
do
    for cr in ${cr_list[@]}
    do
        for channels in "${channels_list[@]}"
        do
            for n_conv in "${n_conv_per_layer[@]}"
            do
                for pool in "${pooling[@]}"
                do
                    for dropout in "${dropout_list[@]}"
                    do
                        for kernel in "${kernel_list[@]}"
                        do
                            for lr in "${learning_rate_list[@]}"
                            do
                                for final_act in "${use_final_act_func[@]}"
                                do
                                    IFS=',' read -r pred_weight weighted_weight grad_weight max_position_weight max_value_weight min_max_position_weight min_max_value_weight fft_weight <<< "$weights"

                                    echo "(pred, weighted pred, grad, max position, max value, inflection value, inflection pos, fft) weights: $weights"

                                    HYDRA_FULL_ERROR=1
                                    python /Odyssey/private/o23gauvr/code/FASCINATION/main.py \
                                    root_save_dir=$root_dir \
                                    save_dir_name=$save_dir \
                                    trainer.max_epochs=$max_epoch \
                                    model.opt_fn.lr=$lr \
                                    model_config.model_hparams.$model.channels_list=$channels \
                                    model_config.model_hparams.$model.n_conv_per_layer=$n_conv \
                                    model_config.model_hparams.$model.pooling=$pool \
                                    model_config.model_hparams.$model.linear_layer.use=True \
                                    model_config.model_hparams.$model.linear_layer.cr=$cr \
                                    model_config.model_hparams.$model.kernel_list=$kernel \
                                    model_config.model_hparams.$model.dropout_proba=$dropout \
                                    model_config.model_hparams.$model.use_final_act_fn=$final_act \
                                    model.loss_weight.prediction_weight=$pred_weight \
                                    model.loss_weight.weighted_weight=$weighted_weight \
                                    model.loss_weight.gradient_weight=$grad_weight \
                                    model.loss_weight.max_position_weight=$max_position_weight \
                                    model.loss_weight.max_value_weight=$max_value_weight \
                                    model.loss_weight.min_max_position_weight=$min_max_position_weight \
                                    model.loss_weight.min_max_value_weight=$min_max_value_weight \
                                    model.loss_weight.fft_weight=$fft_weight \
                                    hydra.job.env_set.CUDA_VISIBLE_DEVICES=$cuda
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Job finished."



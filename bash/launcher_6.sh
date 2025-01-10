#!/bin/bash


declare -a cr_list=(100000 10000 1000) #  #[1,8] 
declare -a loss_weights_list=("1,0,0,0,0,0,0,0")


declare -a pooling_dim="all"
declare -a norm_stats="mean_std"
declare -a interp_size=5

declare -a current_branch=$(git rev-parse --abbrev-ref HEAD)
declare -a cuda=0
declare -a max_epoch=100
declare -a root_dir="/Odyssey/private/o23gauvr/outputs"
declare -a save_dir="'$current_branch/linear_impact_100_epochs'" 

#skip_first=true

# kinit -l 5d
# krenew -K 10 &



for weights in "${loss_weights_list[@]}"
do

    for cr in ${cr_list[@]}
    do

        IFS=',' read -r pred_weight weighted_weight grad_weight max_position_weight max_value_weight min_max_position_weight min_max_value_weight fft_weight <<< "$weights"

        echo "(pred, weighted pred, grad, max position, max value, inflection value, inflection pos, fft) weights: $weights"

        HYDRA_FULL_ERROR=1
        python main.py \
        root_save_dir=$root_dir \
        save_dir_name=$save_dir \
        trainer.max_epochs=$max_epoch \
        model_config.model_hparams.AE_CNN_3D.channels_list=[1,2,4,8] \
        model_config.model_hparams.AE_CNN_3D.n_conv_per_layer=2 \
        model_config.model_hparams.AE_CNN_3D.pooling:"conv" \
        model_config.model_hparams.AE_CNN_3D.linear_layer.use=True \
        model_config.model_hparams.AE_CNN_3D.linear_layer.cr=$ \
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






# if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
#     kinit -l 5d
#     krenew -K 10 &
# fi


            # if [ "$skip_first" = true ]; then
            #     skip_first=false
            #     continue
            # fi

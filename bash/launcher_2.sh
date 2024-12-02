#!/bin/bash


declare -a channels_list=([1,8]) #[1,8] [1,8,8,8] [1,8,8,8,8]
declare -a loss_weights_list=("0.01,1,10000,0,0,0.1,0.5,0" "0.01,1,10000,0,0,0.01,0.5,0" "0.01,1,10000,0.01,0.5,0,0.5,0" "0.01,1,10000,0.1,0.5,0,0,0" "0.01,1,10000,0.1,0.1,0,0,0" "0.01,1,10000,0.01,0.05,0.01,0.05,0.05" "1,0,0,0,0,0,0,0" "0.5,1,0,0,0,0,0,0" "0,1,0,0,0,0,0,0" "1,0,10000,0,0,0,0,0" "0,0,1,0,0,0,0,0" "1,0,0,0,0,0.01,1,0" "0,0,0,0,0,1,1,0" "1,0,0,0.01,1,0,0,0" "0,0,0,1,1,0,0,0" "1,0,0,0,0,0,0,0.1" "0,0,0,0,0,0,0,1" )  #"0.01,1,10000,0.01,0.05,0.01,0.05,0.0000" "1,0,10000,0,0,0,0,0"


declare -a pooling_dim="spatial"
declare -a norm_stats="mean_std"
declare -a pre_treatment_method="none" #none #pca
declare -a cuda=1
declare -a max_epoch=40
declare -a root_dir="/DATASET/envs/o23gauvr/outputs"
declare -a save_dir="'test_on_loss_weights'" 

#skip_first=true

kinit -l 5d
krenew -K 10 &




for chnls in ${channels_list[@]}
do

    for weights in "${loss_weights_list[@]}"
    do
        IFS=',' read -r pred_weight weighted_weight grad_weight max_position_weight max_value_weight inflection_pos_weight inflection_value_weight fft_weight <<< "$weights"


        echo "Channels: $chnls"
        echo "(pred, weighted pred, grad, max position, max value, inflection value, inflection pos, fft) weights: $weights"



        HYDRA_FULL_ERROR=1
        python main.py \
        root_save_dir=$root_dir \
        save_dir_name=$save_dir \
        trainer.max_epochs=$max_epoch \
        datamodule.norm_stats.method=$norm_stats \
        datamodule.depth_pre_treatment.method=$pre_treatment_method \
        datamodule.depth_pre_treatment.params=107 \
        model_config.model_hparams.AE_CNN_3D.channels_list=$chnls \
        model_config.model_hparams.AE_CNN_3D.n_conv_per_layer=1 \
        model_config.model_hparams.AE_CNN_3D.padding=reflect \
        model_config.model_hparams.AE_CNN_3D.interp_size=0 \
        model_config.model_hparams.AE_CNN_3D.upsample_mode=trilinear \
        model_config.model_hparams.AE_CNN_3D.pooling_dim=$pooling_dim \
        model_config.model_hparams.AE_CNN_3D.pooling=Max \
        model_config.model_hparams.AE_CNN_3D.act_fn_str=Elu \
        model_config.model_hparams.AE_CNN_3D.final_act_fn_str=Linear \
        model.opt_fn.lr=0.001 \
        model.loss_weight.prediction_weight=$pred_weight \
        model.loss_weight.weighted_weight=$weighted_weight \
        model.loss_weight.gradient_weight=$grad_weight \
        model.loss_weight.max_position_weight=$max_position_weight \
        model.loss_weight.max_value_weight=$max_value_weight \
        model.loss_weight.inflection_pos_weight=$inflection_pos_weight \
        model.loss_weight.inflection_value_weight=$inflection_value_weight \
        model.loss_weight.fft_weight=$fft_weight \
        hydra.job.env_set.CUDA_VISIBLE_DEVICES=$cuda \




    done
done

# for weights in "${loss_weights_list[@]}"
# do

#     IFS=',' read -r pred_weight weighted_weight grad_weight max_position_weight max_value_weight inflection_pos_weight inflection_value_weight fft_weight <<< "$weights"


#     HYDRA_FULL_ERROR=1
#     python main.py \
#     root_save_dir=$root_dir \
#     save_dir_name=$save_dir \
#     trainer.max_epochs=$max_epoch \
#     datamodule.norm_stats.method=$norm_stats \
#     datamodule.depth_pre_treatment.method=$pre_treatment_method \
#     datamodule.depth_pre_treatment.params=107 \
#     model_config.model_hparams.AE_CNN_3D.channels_list='[1,8]' \
#     model_config.model_hparams.AE_CNN_3D.n_conv_per_layer=1 \
#     model_config.model_hparams.AE_CNN_3D.padding=reflect \
#     model_config.model_hparams.AE_CNN_3D.interp_size=0 \
#     model_config.model_hparams.AE_CNN_3D.upsample_mode=trilinear \
#     model_config.model_hparams.AE_CNN_3D.pooling_dim=all \
#     model_config.model_hparams.AE_CNN_3D.pooling=None \
#     model_config.model_hparams.AE_CNN_3D.act_fn_str=Elu \
#     model_config.model_hparams.AE_CNN_3D.final_act_fn_str=Linear \
#     model.opt_fn.lr=0.001 \
#     model.loss_weight.prediction_weight=$pred_weight \
#     model.loss_weight.weighted_weight=$weighted_weight \
#     model.loss_weight.gradient_weight=$grad_weight \
#     model.loss_weight.max_position_weight=$max_position_weight \
#     model.loss_weight.max_value_weight=$max_value_weight \
#     model.loss_weight.inflection_pos_weight=$inflection_pos_weight \
#     model.loss_weight.inflection_value_weight=$inflection_value_weight \
#     model.loss_weight.fft_weight=$fft_weight \
#     hydra.job.env_set.CUDA_VISIBLE_DEVICES=$cuda \
    
# done






# if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
#     kinit -l 5d
#     krenew -K 10 &
# fi


            # if [ "$skip_first" = true ]; then
            #     skip_first=false
            #     continue
            # fi

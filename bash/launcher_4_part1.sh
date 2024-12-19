
#!/bin/bash


declare -a channels_list=([1,8] [1,8,8])
declare -a loss_weights_list=("0.5,1,0,0.1,0.5,0,0,0" "1,0,0,0,0,0,0,0")
declare -a pca_components=(1 3 5) 
declare -a norm_stats_list=("mean_std_along_depth" "mean_std" )
declare -a pre_treatment_norm_on=("components" "profiles")
declare -a pre_treatment_train_on=("components" "profiles")

declare -a pooled_dim="spatial"
#declare -a norm_stats="mean_std_along_depth" #mean_std #min_max #mean_std_along_depth
declare -a pre_treatment_method="pca" #none #pca
# declare -a pre_treatment_norm_on="components" #profiles #components
# declare -a pre_treatment_train_on="components"  #components #profiles

declare -a current_branch=$(git rev-parse --abbrev-ref HEAD)
declare -a cuda=0
declare -a max_epoch=5
declare -a root_dir="/Odyssey/private/o23gauvr/outputs"
declare -a save_dir="'$current_branch/pca_pre_treatment_5_epochs'" 

#skip_first=true

kinit -l 5d
krenew -K 10 &




for pca in ${pca_components[@]}
do

    for chnls in ${channels_list[@]}
    do


        for norm_stats in ${norm_stats_list[@]}
        do

            for pre_treatment_norm_on in ${pre_treatment_norm_on[@]}
            do

                for pre_treatment_train_on in ${pre_treatment_train_on[@]}
                do

                    for weights in "${loss_weights_list[@]}"
                    do
                        IFS=',' read -r pred_weight weighted_weight grad_weight max_position_weight max_value_weight min_max_position_weight min_max_value_weight fft_weight <<< "$weights"



                        echo "Channels: $chnls"
                        echo "(pred, weighted pred, grad, max position, max value, inflection value, inflection pos, fft) weights: $weights"
                        echo "PCA components: $pca"
                        echo "Norm stats: $norm_stats"
                        echo "Pre-treatment method: $pre_treatment_method"
                        echo "Pre-treatment norm on: $pre_treatment_norm_on"



                        HYDRA_FULL_ERROR=1
                        python main.py \
                        root_save_dir=$root_dir \
                        save_dir_name=$save_dir \
                        pooled_dim=$pooled_dim \
                        trainer.max_epochs=$max_epoch \
                        datamodule.norm_stats.method=$norm_stats \
                        datamodule.depth_pre_treatment.method=$pre_treatment_method \
                        datamodule.depth_pre_treatment.params=$pca \
                        datamodule.depth_pre_treatment.norm_on=$pre_treatment_norm_on \
                        datamodule.depth_pre_treatment.train_on=$pre_treatment_train_on \
                        model_config.model_hparams.AE_CNN_3D.channels_list=$chnls \
                        model_config.model_hparams.AE_CNN_3D.n_conv_per_layer=1 \
                        model_config.model_hparams.AE_CNN_3D.padding=reflect \
                        model_config.model_hparams.AE_CNN_3D.interp_size=0 \
                        model_config.model_hparams.AE_CNN_3D.upsample_mode=trilinear \
                        model_config.model_hparams.AE_CNN_3D.pooling=Max \
                        model_config.model_hparams.AE_CNN_3D.act_fn_str=Elu \
                        model_config.model_hparams.AE_CNN_3D.final_act_fn_str=Linear \
                        model.opt_fn.lr=0.001 \
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
                done
            done
        done
    done
done



# for weights in "${loss_weights_list[@]}"
# do

#     IFS=',' read -r pred_weight weighted_weight grad_weight max_position_weight max_value_weight min_max_position_weight min_max_value_weight fft_weight <<< "$weights"


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
#     model.loss_weight.min_max_position_weight=$min_max_position_weight \
#     model.loss_weight.min_max_value_weight=$min_max_value_weight \
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
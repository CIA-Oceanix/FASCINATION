#!/bin/bash

final_act_fn_str_values=("None")
act_fn_str_values=("None" "Relu" "Elu" "Sigmoid")


declare -a current_branch=$(git rev-parse --abbrev-ref HEAD)
declare -a cuda=0
declare -a max_epoch=5
declare -a root_dir="/Odyssey/private/o23gauvr/outputs"
declare -a save_dir="'$current_branch/test_on_act_fn'" 

kinit -l 5d
krenew -K 10 &


# Loop over the values of final_act_fn_str and act_fn_str
for final_act_fn_str in "${final_act_fn_str_values[@]}"
do
  for act_fn_str in "${act_fn_str_values[@]}"
  do

    echo "Running with final_act_fn_str=${final_act_fn_str} and act_fn_str=${act_fn_str}"

    HYDRA_FULL_ERROR=1 \

    python main.py \
    root_save_dir=$root_dir \
    save_dir_name=$save_dir \
    pooled_dim="all" \
    trainer.max_epochs=$max_epoch \
    hydra.job.env_set.CUDA_VISIBLE_DEVICES=$cuda \
    model_config.model_hparams.AE_CNN_3D.channels_list=[1,8] \
    model_config.model_hparams.AE_CNN_3D.final_act_fn_str=${final_act_fn_str}\
    model_config.model_hparams.AE_CNN_3D.act_fn_str=${act_fn_str}\



  done
done
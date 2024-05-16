#!/bin/bash

declare -a architecture=("dense_3D_CNN_ReLu_1st_kernel_size" "dense_3D_CNN_ReLu")
declare -a loss_weights=( "0,1" "1,0" "1,1" "10,1" "100,1" "1000,1" "1,10" "1,100" "10000,1")

# Initialize Kerberos ticket with a lifetime of 3 days
kinit -l 3d

for arch in "${architecture[@]}"
do
    echo "architecture: '$arch'"

    if [ "$arch" == "dense 3D CNN ReLu" ]; then
        for weights in "${loss_weights[@]}"
        do
            IFS=',' read -r classif_weight pred_weight <<< "$weights"
            echo "classification weight: '$classif_weight'"
            echo "prediction weight: '$pred_weight'"

            HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py model.arch_shape="'$arch'" model.classif_weight=$classif_weight model.pred_weight=$pred_weight
        done
    else
        HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python main.py model.arch_shape="'$arch'"
    fi
done


#!/bin/bash

declare -a architecture=("dense_3D_CNN_ReLu" "dense_3D_CNN_ReLu_2nd_kernel_size" 
                        "dense_3D_CNN_ReLu_kernel_size_5" "dense_3D_CNN_ReLu_kernel_size_8" "dense_3D_CNN_ReLu_kernel_size_20" 
                        "dense_2D_CNN_ReLu" "lucas_model")
declare -a loss_weights=("1,100" "10000,1") #"0,1"

# Initialize Kerberos ticket with a lifetime of 3 days
kinit -l 5d

for arch in "${architecture[@]}"
do
    echo "architecture: '$arch'"

    if [ "$arch" == "dense_3D_CNN_ReLu" ]; then
        for weights in "${loss_weights[@]}"
        do
            IFS=',' read -r classif_weight pred_weight <<< "$weights"
            echo "classification weight: '$classif_weight'"
            echo "prediction weight: '$pred_weight'"

            HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=4 python main.py model.arch_shape="'$arch'" model.classif_weight=$classif_weight model.pred_weight=$pred_weight
        done
    else
        HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=4 python main.py model.arch_shape="'$arch'"
    fi
done


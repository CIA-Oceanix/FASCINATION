#!/bin/bash

declare -a architecture=("4_15" "4_15_test" "pca_4")
declare -a active_func=("relu" "sigmoid")
declare -a lr_rate=(0.001 0.01 0.1)
#gpu = (1,2)

# for i in ${#arch[@]}
#for arch in ${architecture[@]}
kinit -l 1h -r 6d 
for arch in "${architecture[@]}"
do
    for func in "${active_func[@]}"
    do
        for lr in "${lr_rate[@]}"
        do
            # echo $arch
            # echo "$arch"   ####Similaire à echo $arch
            echo "architecture: '$arch'"
            echo "activation function: '$func'"
            echo "learning rate: $lr"
            #echo model.arch_shape="$arch"
            HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=4 python main.py xp=AE_without_AP_training model.arch_shape="'$arch'" model.final_act_func="'$func'" model.lr=$lr 

            # --cfg job
        done
    done
done

#!/bin/bash

declare -a architecture=("4_15" "16_60" "32_120"  "8_30" "no_pool_4")
#gpu = (1,2)

# for i in ${#arch[@]}
#for arch in ${architecture[@]}
kinit -R 
for arch in "${architecture[@]}"
do
    
    # echo $arch
    # echo "$arch"   ####Similaire à echo $arch
    echo "architecture: '$arch'" #####Ecriture correcte
    #echo model.arch_shape="$arch"
    CUDA_VISIBLE_DEVICES=2 python main.py xp=AE_without_AP_training model.arch_shape="'$arch'"
    #CUDA_VISIBLE_DEVICES=1 krenew python main.py xp=AE_with_AP_training model.arch_shape="'$arch'"

done
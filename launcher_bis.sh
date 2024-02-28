#!/bin/bash

declare -a architecture=("32_120" "16_60")
#gpu = (1,2)

# for i in ${#arch[@]}
#for arch in ${architecture[@]}
for arch in "${architecture[@]}"
do
    # echo $arch
    # echo "$arch"   ####Similaire à echo $arch
    echo "architecture: '$arch'" #####Ecriture correcte
    #echo model.arch_shape="$arch"
    CUDA_VISIBLE_DEVICES=1 krenew python main.py model.arch_shape="'$arch'"
    # echo "$gpu[$i]"

done
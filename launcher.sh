#!/bin/bash

declare -a architecture=("4_60_dense_avgpool_upsample_no_activation" "20_60_dense_avgpool_upsample_no_activation"
                        "4_60_dense_avgpool_upsample_all_Relu" "20_60_dense_avgpool_upsample_all_Relu"
                        "4_60_dense_avgpool_upsample_final_Relu" "20_60_dense_avgpool_upsample_final_Relu" 
                        "no_pool_20" "no_pool_4"
                        "8_30" "16_60" "32_120")

# declare -a active_func=("relu")
# declare -a lr_rate=(0.001)
#gpu = (1,2)

# for i in ${#arch[@]}
#for arch in ${architecture[@]}
#kinit -l 1h -r 6d 
kinit -l 3d
for arch in "${architecture[@]}"
do
    echo "architecture: '$arch'"
    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python main.py xp=AE_without_AP_training model.arch_shape="'$arch'"

done


# do
#     for func in "${active_func[@]}"
#     do
#         for lr in "${lr_rate[@]}"
#         do
#             # echo $arch
#             # echo "$arch"   ####Similaire à echo $arch
#             echo "architecture: '$arch'"
#             echo "activation function: '$func'"
#             echo "learning rate: $lr"
#             #echo model.arch_shape="$arch"
#             HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python main.py xp=AE_with_AP_training model.arch_shape="'$arch'" model.final_act_func="'$func'" model.lr=$lr 

#             # --cfg job
#         done
#     done
# done
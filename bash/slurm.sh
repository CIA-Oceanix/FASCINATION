#!/bin/bash -l
#SBATCH --partition=Odyssey_GPU            
#SBATCH --job-name=(ᵔ+ᵔ #Metrics #(ꈍ-ꈍ  #(ᵔ+ᵔ #(ꈍ◡ꈍ #OwO #UwU #n_n #n_n  #OwO  ~_~  UwU é_è  .__.  n_n
#SBATCH --gres=gpu:h200:1  ####   #SBATCH --exclude=sl-mee-br-212
#SBATCH --mem=150G
#SBATCH --cpus-per-task=20
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            

echo "Job started."

source /Odyssey/private/o23gauvr/start_conda.sh

#export CONDARC=/Odyssey/private/o23gauvr/miniforge3/.condarc 
#conda infos
source activate fsc #fsc #fsc_test #fsc_env #run_model
echo "Environment activated successfully."



#bash /Odyssey/private/o23gauvr/code/FASCINATION/bash/dir_size_tree.sh /Odyssey/private/o23gauvr /Odyssey/private/o23gauvr/code/all_memory_use.log
HYDRA_FULL_ERROR=1 srun python /Odyssey/private/o23gauvr/code/FASCINATION/experiments/residual_flow_full_mlic/src/residual_flow.py #/Odyssey/private/o23gauvr/code/FASCINATION/main.py #/Odyssey/private/o23gauvr/code/MLIC/MLIC/playground/train.py


echo "Job finished."



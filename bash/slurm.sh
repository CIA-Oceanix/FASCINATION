#!/bin/bash -l
#SBATCH --partition=Odyssey            
#SBATCH --job-name=metrics #UwU #(ꈍ-ꈍ  #(ᵔ+ᵔ #(ꈍ◡ꈍ #OwO #UwU #n_n #n_n  #OwO  ~_~  UwU é_è  .__.  n_n
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=70
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            

echo "Job started."

source /Odyssey/private/o23gauvr/start_conda.sh

#export CONDARC=/Odyssey/private/o23gauvr/miniforge3/.condarc 
#conda infos
source activate fsc #fsc #fsc_test #fsc_env #run_model
echo "Environment activated successfully."



HYDRA_FULL_ERROR=1 srun python /Odyssey/private/o23gauvr/code/FASCINATION/src/full_metrics.py #/Odyssey/private/o23gauvr/code/FASCINATION/main.py #/Odyssey/private/o23gauvr/code/MLIC/MLIC/playground/train.py
echo "Job finished."



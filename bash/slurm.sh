#!/bin/bash -l
#SBATCH --partition=Odyssey            
#SBATCH --job-name=UwU  
#SBATCH --gres=gpu:l40s:1       
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            



echo "Job started."


source /Odyssey/private/o23gauvr/start_conda.sh

#export CONDARC=/Odyssey/private/o23gauvr/miniforge3/.condarc
conda info

source activate run_model
echo "Environment activated successfully."


HYDRA_FULL_ERROR=1 srun python /Odyssey/private/o23gauvr/code/FASCINATION/main.py \



echo "Job finished."



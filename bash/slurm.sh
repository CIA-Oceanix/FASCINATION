#!/bin/bash -l
#SBATCH --partition=Odyssey            
#SBATCH --job-name=MLIC 
#SBATCH --gres=gpu:h100:1       
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            



echo "Job started."


source /Odyssey/private/o23gauvr/start_conda.sh

#export CONDARC=/Odyssey/private/o23gauvr/miniforge3/.condarc
conda info
source activate fsc_env #run_model
echo "Environment activated successfully."


HYDRA_FULL_ERROR=1 srun python /Odyssey/private/o23gauvr/code/MLIC/MLIC/playground/train.py


echo "Job finished."



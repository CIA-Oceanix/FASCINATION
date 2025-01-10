#!/bin/bash
#SBATCH --partition=Odyssey            
#SBATCH --job-name=TestEnvSetup  
#SBATCH --gres=gpu:h100:1              
#SBATCH --output=test_job_%j.log            
#SBATCH --error=test_job_%j.err

module load Anaconda3/2020.07
kinit -l 5d
krenew -K 10 &
source ~/.bashrc
conda activate fsc_env
echo "Environment activated successfully."

bash /homes/o23gauvr/Documents/thèse/code/FASCINATION/bash/launcher_6.sh

echo "Job finished."



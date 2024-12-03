#!/bin/bash
#SBATCH --partition=Odyssey            
#SBATCH --job-name=TestEnvSetup  
#SBATCH --gres=gpu:h100:1              
#SBATCH --output=test_job_%j.log            
#SBATCH --error=test_job_%j.err

source ~/.bashrc
conda activate fsc_env
echo "Environment activated successfully."

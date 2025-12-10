#!/bin/bash -l
#SBATCH --partition=Odyssey            
#SBATCH --job-name=metrics #UwU #n_n #n_n  #OwO  ~_~  UwU é_è  .__.  n_n
#SBATCH --gres=gpu:l40s:1       
#SBATCH --mem=150G
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_%j.log            

echo "Job started."

source /Odyssey/private/o23gauvr/start_conda.sh

#export CONDARC=/Odyssey/private/o23gauvr/miniforge3/.condarc
#conda infos
source activate fsc #fsc #fsc_test #fsc_env #run_model
echo "Environment activated successfully."



HYDRA_FULL_ERROR=1 srun python /Odyssey/private/o23gauvr/code/FASCINATION/src/pca_compo_dict.py #/Odyssey/private/o23gauvr/code/FASCINATION/src/compute_metrics.py #/Odyssey/private/o23gauvr/code/MLIC/MLIC/playground/train.py #/Odyssey/private/o23gauvr/code/FASCINATION/main.py # #/Odyssey/private/o23gauvr/code/FASCINATION/src/compute_metrics.py  #/Odyssey/private/o23gauvr/code/FASCINATION/src/compute_model_metrics.py/Odyssey/private/o23gauvr/code/FASCINATION/src/jpeg2000.py #/Odyssey/private/o23gauvr/code/FASCINATION/src/udate_pca_per_season.py # #/Odyssey/private/o23gauvr/code/MLIC/MLIC/playground/train.py #/Odyssey/private/o23gauvr/code/MLIC/MLIC/playground/train.py


echo "Job finished."



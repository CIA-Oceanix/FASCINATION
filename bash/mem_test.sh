#!/bin/bash
#SBATCH --job-name=mem_test
#SBATCH --partition=Odyssey            
#SBATCH --gres=gpu:h100:1       
#SBATCH --output=/Odyssey/private/o23gauvr/code/FASCINATION/logs/mem_test_150_h100.out
#SBATCH --error=/Odyssey/private/o23gauvr/code/FASCINATION/logs/mem_test_150_h100.err
#SBATCH --mem=160G  # Adjust this value

echo "Starting job on $(hostname)"
free -h
cat /proc/meminfo | grep Mem
python3 -c "a = ' ' * (1024**3 * 150)"  # Try to allocate 150 GB
echo "Memory test 150G finished"
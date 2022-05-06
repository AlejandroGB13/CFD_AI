#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 160
# Number of GPUs per node
#SBATCH --gres=gpu:0
#SBATCH --time 00-20:00:00
#SBATCH --exclusive
#SBATCH --job-name="train_tensorflow_cpu"
#SBATCH -o "%x-%j.out"

python script_tensorflow.py

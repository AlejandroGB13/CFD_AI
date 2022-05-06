#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

srun python script_multiworker.py

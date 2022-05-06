#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks 160
# Number of GPUs per node
#SBATCH --gres=gpu:0
#SBATCH --time 00-24:00:00
#SBATCH --exclusive
#SBATCH --job-name="train_hvd_cpu"
#SBATCH -o "%x-%j.out"

nodes=$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)
python script_gpu_v3.py $nodes 2
variable=$(head -1 script_result_v3.txt)
echo $variable
horovodrun -np 2 -H $variable python script_horovod_cpu.py
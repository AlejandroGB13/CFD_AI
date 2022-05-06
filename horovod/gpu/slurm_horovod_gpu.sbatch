#!/bin/bash
#SBATCH --nodes 4
#SBATCH --ntasks 160
# Number of GPUs per node
#SBATCH --gres=gpu:1
#SBATCH --time 00-03:00:00
#SBATCH --exclusive
#SBATCH --job-name="train_hvd_gpu"
#SBATCH -o "%x-%j.out"

nodes=$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)
python script_gpu_v3.py $nodes 4
variable=$(head -1 script_result_v3.txt)
echo $variable
horovodrun -np 4 -H $variable python script_horovod_gpu.py
#$ horovodrun -np 4 --timeline-filename /path/to/timeline.json python train.py
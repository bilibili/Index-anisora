#!/bin/bash
#SBATCH --job-name=test_remote_data
#SBATCH --output=test_remote_data_%j.out
#SBATCH --error=test_remote_data_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=gpu
#SBATCH --export=ALL
#SBATCH --gres=gpu:8

srun -l single_launch.sh 
echo "Done with job $SLURM_JOB_ID"

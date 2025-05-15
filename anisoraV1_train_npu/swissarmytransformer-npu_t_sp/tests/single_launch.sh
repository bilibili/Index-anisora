#!/bin/bash
# author: mingding

# this is launched by srun
# command for this script: srun -N 2 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=4 --job-name=slurm_example --partition=dev --time=00:10:00 --output=slurm_example.out --error=slurm_example.err ./single_launch.sh

# if SLURM defined, set by SLURM environment
WORLD_SIZE=${SLURM_NTASKS:-1}
RANK=${SLURM_PROCID:-0}
# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$SLURM_NODELIST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=7878
else
    MASTER_ADDR=`scontrol show hostnames $SLURM_NODELIST | head -n 1`
    MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi
# generate a port at random
LOCAL_RANK=${SLURM_LOCALID:-0}

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


# export S3_ENDPOINT_URL=
# export S3_ACCESS_KEY_ID=
# export S3_SECRET_ACCESS_KEY=

python test_remote_data.py --world_size $WORLD_SIZE --rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --local_rank $LOCAL_RANK

echo "DONE on `hostname`"
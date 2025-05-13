#! /bin/bash

#module load cuda
#echo "RUN on `localhost`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"
RANDOM=1024
run_cmd="$environs python train_video.py --base configs/cogvideox2b_lora.yaml configs/finetune_config.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

#echo "DONE on `localhost`"
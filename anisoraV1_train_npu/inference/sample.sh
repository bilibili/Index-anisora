#! /bin/bash
hostname="localhost"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video.py --base configs/base_video.yaml configs/cogvideox5b.yaml  --force-inference"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
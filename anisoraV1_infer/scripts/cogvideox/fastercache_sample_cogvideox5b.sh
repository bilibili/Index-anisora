#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python scripts/cogvideox/fastercache_sample_cogvideox.py --base configs/cogvideox/fastercache_sample_5b.yaml"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"


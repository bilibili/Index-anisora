#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd="python scripts/cogvideox/fastercache_sample_cogvideox_sp.py --base configs/cogvideox/fastercache_sample_5b.yaml"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"

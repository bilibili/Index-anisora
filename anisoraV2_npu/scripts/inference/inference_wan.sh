#!/bin/bash

PROJECT_NAME=wan-infer
DATA="$(date +"%m%d%H%M")"

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
cd $SCRIPT_ROOT/../..

export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

#rm -rf data/outputs/${PROJECT_NAME}

torchrun --nnodes 1 \
         --nproc_per_node 16 \
         fastvideo/bili_space/wan/generate-pi-i2v.py \
         --base_seed 42 \
         --task i2v-14B \
         --size 832*480 \
         --ckpt_dir Wan2.1-I2V-14B-480P \
         --image data/outputs/${PROJECT_NAME} \
         --prompt test_data/todo-inference.txt \
         --dit_fsdp \
         --t5_fsdp \
         --ulysses_size 1 --frame_num 49


#!/bin/bash

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
cd $SCRIPT_ROOT/../..

# export VS_DEBUG=True

# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0077.html
export HCCL_INTRA_ROCE_ENABLE=1
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0079.html
export HCCL_INTRA_PCIE_ENABLE=0
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0122.html
# export ASCEND_GLOBAL_EVENT_ENABLE=1
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0120.html
# export ASCEND_GLOBAL_LOG_LEVEL=1
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0091.html
# export HCCL_ENTRY_LOG_ENABLE=1
# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0050.html
# export ASCEND_LAUNCH_BLOCKING=1
# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0074.html
# export HCCL_CONNECT_TIMEOUT=300
# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0075.html
# export HCCL_EXEC_TIMEOUT=300
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0053.html
# export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:2048,garbage_collection_threshold:0.6,expandable_segments:False
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

SP_SIZE=8
num_gpus=$SP_SIZE
model_base=data/hunyuan
dit_weight=${model_base}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
# dit_weight=/workspace/fastvideo/checkpoint-1000/diffusion_pytorch_model.safetensors

export MODEL_BASE=${model_base}

torchrun --nnodes=1 \
         --nproc_per_node=$num_gpus \
         --master_addr=127.0.0.1 \
         --master_port 10000 \
         fastvideo/sample/sample_t2v_hunyuan.py \
         --height 720 \
         --width 1280 \
         --num_frames 93 \
         --num_inference_steps 12 \
         --guidance_scale 1 \
         --embedded_cfg_scale 6 \
         --flow_shift 17 \
         --flow-reverse \
         --seed 1024 \
         --output_path outputs_video/ \
         --model_path $MODEL_BASE \
         --prompt fastvideo/bili_space/prompt.txt \
         --dit-weight $dit_weight

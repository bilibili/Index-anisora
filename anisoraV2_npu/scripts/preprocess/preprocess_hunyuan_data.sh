#!/bin/bash

PROJECT_NAME=bili-Hunyuan-train-data
DATA="$(date +"%m%d%H%M")"

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
cd $SCRIPT_ROOT/../..

# export VS_DEBUG=True

# export BOSS_CKPT_ENDPOINT=http://jssz-inner-boss.bilibili.co
# export BOSS_CKPT_ACCESS=<REDACTED_SECRET>
# export BOSS_CKPT_SECRET=<REDACTED_SECRET>
# export BOSS_CKPT_BUCKET=virtualhuman
# export BOSS_CKPT_PATH=2D/tmp
# export BOSS_CKPT_PATH=${BOSS_CKPT_PATH}/${PROJECT_NAME}/${DATA}

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

# export WANDB_MODE="offline"
GPU_NUM=16
MODEL_PATH="data/hunyuan"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="data/Image-Vid-Finetune-Src/merge.txt"
OUTPUT_DIR="data/Image-Vid-Finetune-HunYuan"
VALIDATION_PATH="assets/prompt.txt"
# TEXT_ENCODER_NAME="data/hunyuan/text_encoder"

rm -rf $OUTPUT_DIR

torchrun --nnodes 1 \
         --nproc_per_node=$GPU_NUM \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=10000 \
         fastvideo/data_preprocess/preprocess_vae_latents.py \
         --model_path $MODEL_PATH \
         --data_merge_path $DATA_MERGE_PATH \
         --train_batch_size=1 \
         --max_height=720 \
         --max_width=1280 \
         --num_frames=93 \
         --dataloader_num_workers 1 \
         --output_dir=$OUTPUT_DIR \
         --model_type $MODEL_TYPE \
         --train_fps 24

torchrun --nnodes 1 \
         --nproc_per_node=$GPU_NUM \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=10000 \
         fastvideo/data_preprocess/preprocess_text_embeddings.py \
         --model_type $MODEL_TYPE \
         --model_path $MODEL_PATH \
         --output_dir=$OUTPUT_DIR

torchrun --nnodes 1 \
         --nproc_per_node=1 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=10000 \
         fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
         --model_type $MODEL_TYPE \
         --model_path $MODEL_PATH \
         --output_dir=$OUTPUT_DIR \
         --validation_prompt_txt $VALIDATION_PATH

#!/bin/bash

PROJECT_NAME=wan-finetune
DATA="$(date +"%m%d%H%M")"

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
cd $SCRIPT_ROOT/../..

export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

# Check deterministic
export DETERMINISTIC=false
if [[ "${DETERMINISTIC,,}" == "true" ]]; then
    echo "DETERMINISTIC is true (case insensitive)"
    if [[ "${ACCELERATOR,,}" == "npu" ]]; then
        export HCCL_DETERMINISTIC=true
        export CLOSE_MATMUL_K_SHIFT=1
    else
        :
    fi
else
    echo "DETERMINISTIC is not true"
fi

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline

rm -rf kernel_meta npu_prof_result

torchrun --nnodes 1 \
         --nproc_per_node 16 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=10000 \
         fastvideo/train.py \
         --seed 42 \
         --pretrained_model_name_or_path Wan2.1-I2V-14B-480P \
         --dit_model_name_or_path Wan2.1-I2V-14B-480P \
         --model_type wan \
         --cache_dir data/.cache \
         --data_json_path test_data/data.json \
         --validation_prompt_dir data/Image-Vid-Finetune-HunYuan/validation \
         --master_weight_type fp32 \
         --fsdp_sharding_startegy full \
         --train_batch_size=1 \
         --num_latent_t 24 \
         --sp_size 2 \ # sp_size max = 8
         --train_sp_batch_size 1 \
         --dataloader_num_workers 4 \
         --gradient_accumulation_steps=1 \
         --max_train_steps=9000 \
         --learning_rate=2e-5 \
         --betas 0.9 0.95 \
         --mixed_precision=bf16 \
         --checkpointing_steps=3 \
         --validation_steps 100 \
         --validation_sampling_steps 16 \
         --checkpoints_total_limit 3 \
         --allow_tf32 \
         --ema_start_step 0 \
         --cfg 0.0 \
         --ema_decay 0.999 \
         --output_dir=data/outputs/${PROJECT_NAME} \
         --tracker_project_name ${PROJECT_NAME} \
         --validation_num_frames 93 \
         --validation_width 1280 \
         --validation_height 720 \
         --validation_guidance_scale "1.0" \
         --group_frame

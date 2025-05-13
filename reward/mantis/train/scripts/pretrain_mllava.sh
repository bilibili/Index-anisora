#!/bin/bash
#SBATCH --job-name=train_fuyu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -p a100
#SBATCH --gpus-per-node=4
#SBATCH --time=72:00:00
#SBATCH --qos=a100_wenhuchen
#SBATCH --mem=230GB
#SBATCH --output=../../jobs/%j.out

nvidia-smi
nvcc --version

# offline training
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

if [ "$HF_DATASETS_OFFLINE" = 1 ]; then
    echo "Warning: Offline mode is enabled. Using local copy of datasets"
    DATA_CONFIG_FILE=""
    echo "Please set offline DATA_CONFIG_FILE"
    exit 1
else
    DATA_CONFIG_FILE="./data_configs/llava_pretrain.yaml"
fi
if [ "$TRANSFORMERS_OFFLINE" = 1 ]; then
    echo "Warning: Offline mode is enabled. Using local copy of models"
    echo "Please set offline model path"
    exit 1
else
    vision_backbone="openai/clip-vit-large-patch14-336"
    # vision_backbone="google/siglip-so400m-patch14-384"
    llm_backbone="meta-llama/Meta-Llama-3-8B-Instruct"
    # llm_backbone="lmsys/vicuna-7b-v1.5"
    
fi
if [ "$HF_HUB_OFFLINE" = 1 ]; then
    echo "Warning: Offline mode is enabled. Using local copy of model and datasets"
    push_to_hub=False
else
    push_to_hub=True
fi
if [ -z $HF_HOME ]; then
    echo "HF_HOME is empty, set to default '~/.cache/huggingface/'"
    export HF_HOME="~/.cache/huggingface/"
fi
if [ -z $HF_TOKEN ]; then
    echo "HF token is empty, try loading from '$HF_HOME/token'"
    export HF_TOKEN=$(eval "cat ${HF_HOME}/token")
fi
if [ -z $HF_TOKEN ]; then
    echo "HF token cannot be found, please set your HF token"
    exit 1
fi

hf_hub_user_name="" # set this will push the model to your hub after training
do_pretrain=True
max_seq_len=8192
lora_enabled=false
OUTPUT_DIR="../../checkpoints"
global_batch_size=256
mllava_type="llava"

RUN_NAME="${mllava_type}_clip_llama3_8b_pretrain_8192"
RUN_NAME="${mllava_type}_siglip_llama3_8b_pretrain_8192"
export WANDB_PROJECT="Mantis"
if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    RUN_NAME="${RUN_NAME}_${max_seq_len}_lora"
else
    echo "lora is disabled"
    RUN_NAME="${RUN_NAME}_${max_seq_len}"
fi
echo "RUN_NAME = $RUN_NAME"

hub_model_id="${hf_hub_user_name}/${RUN_NAME}" # the hub model id
hub_token=$HF_TOKEN # set in .bashrc or replace with your own token
if [ -z $hf_hub_user_name ]; then
    echo "hf_hub_user_name is empty, do not push to hub"
    push_to_hub=False
else
    echo "hf_hub_user_name = $hf_hub_user_name"
fi
# resume from checkpoint
resume_from_checkpoint=""
if [ -d $resume_from_checkpoint ]; then
    echo "resume_from_checkpoint = $resume_from_checkpoint"
    export WANDB_LAST_RUN_ID="your_last_run_id"
else
    echo "No checkpoint found, training from scratch"
fi

export NCCL_DEBUG=INFO;
export CXX=g++;

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export COUNT_NODE=$WORLD_SIZE

if [ -z $HOSTNAMES ]; then
    echo "HOSTNAMES is empty"
    export HOSTNAMES=$(hostname | awk '{print $1}')
fi
if [ -z $MASTER_ADDR ]; then
    echo "MASTER_ADDR is empty"
    export MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi
if [ -z $MASTER_PORT ]; then
    echo "MASTER_PORT is empty"
    export MASTER_PORT=12956
fi
if [ -z $COUNT_NODE ]; then
    echo "COUNT_NODE is empty"
    export COUNT_NODE=1
fi
if [ -z $RANK ]; then
    echo "RANK is empty"
    export RANK=0
fi


NGPU_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c "$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')")
GPU=$((${COUNT_NODE} * ${NGPU_PER_NODE}))
WORKERS=$((${COUNT_NODE} * ${NGPU_PER_NODE} * 4))

if [ $WORKERS -gt 112 ]; then
    WORKERS=112
fi

echo HOSTNAMES = $HOSTNAMES
echo hostname = $(hostname)
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
echo GPU=${GPU}
echo COUNT_NODE=$COUNT_NODE
echo WORKERS=$WORKERS
echo "Running ${RUN_NAME}"

# if lora is enabled, please use zero2
# if lora is disabled, please use zero3
if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    echo "Using zero2"
    config_file="./accelerate_configs/accelerate_config_zero2.yaml"
else
    echo "lora is disabled"
    echo "Using zero2"
    config_file="./accelerate_configs/accelerate_config_zero2.yaml"
fi

per_device_train_batch_size=1
gradient_accumulation_steps=$(($global_batch_size / ($per_device_train_batch_size * $GPU)))
echo gradient_accumulation_steps=$global_batch_size / \($per_device_train_batch_size \* $GPU\) = $gradient_accumulation_steps

accelerate launch --config_file=$config_file \
    --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines=${COUNT_NODE} --num_processes=${GPU} \
    train_mllava.py --do_pretrain $do_pretrain \
    --llm_backbone $llm_backbone \
    --vision_backbone $vision_backbone \
    --data_config_file $DATA_CONFIG_FILE \
    --conv_template "plain" \
    --run_name $RUN_NAME \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --hub_model_id $hub_model_id \
    --hub_token $hub_token \
    --push_to_hub $push_to_hub \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --dataloader_num_workers $WORKERS \
    --report_to wandb \
    --do_train \
    --lora_enabled $lora_enabled \
    --max_seq_len $max_seq_len \
    --resume_from_checkpoint "$resume_from_checkpoint"
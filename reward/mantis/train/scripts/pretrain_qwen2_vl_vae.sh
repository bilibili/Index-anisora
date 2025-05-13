nvidia-smi
nvcc --version
# offline training
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

if [ "$HF_DATASETS_OFFLINE" = 1 ]; then
    echo "Warning: Offline mode is enabled. Using local copy of datasets"
    DATA_CONFIG_FILE="./data_configs/train_config_offline.yaml"
else
    # DATA_CONFIG_FILE="./data_configs/mantis_instruct.yaml"  # change to this for offical training
    DATA_CONFIG_FILE="./data_configs/train_config_debug.yaml"
    # DATA_CONFIG_FILE="./data_configs/llava_pretrain.yaml"
    # DATA_CONFIG_FILE="/h/dongfu/WorkSpace/Mantis/data/examples/chat_video/data_config.yaml"
fi
if [ "$TRANSFORMERS_OFFLINE" = 1 ]; then
    echo "Warning: Offline mode is enabled. Using local copy of models"
    model_name_or_path="{local_model_path}"
else
    model_name_or_path="Qwen/Qwen2-VL-7B-Instruct"
    
    # vae_class_name=AutoencoderKLMochi
    # vae_model_name_or_path="genmo/mochi-1-preview"
    # vae_subfolder="vae"
    # vae_latent_channel_size=12
    # vae_temporal_compress_rate=6
    # vae_width_compress_rate=8
    # vae_height_compress_rate=8
    # post_vae_patch_size=2

    vae_class_name=AutoencoderKLCogVideoX
    vae_model_name_or_path="THUDM/CogVideoX-2b"
    vae_subfolder="vae"
    vae_latent_channel_size=16
    vae_temporal_compress_rate=4
    vae_width_compress_rate=8
    vae_height_compress_rate=8
    post_vae_patch_size=2
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

do_pretrain=True
hf_hub_user_name="Mantis-VL" # set this will push the model to your hub after training
max_seq_len=16384
lora_enabled=false
qlora_enabled=false
OUTPUT_DIR="../../checkpoints"
global_batch_size=64
min_pixels=256
max_pixels=640
use_liger_kernel=False

RUN_NAME="mantis-8b-qwen2-vl-vae-pretrain"
export WANDB_PROJECT="Mantis"
if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    if [ $qlora_enabled = true ]; then
        echo "qlora & dora is enabled"
        RUN_NAME="${RUN_NAME}_${max_seq_len}_qlora"
    else
        RUN_NAME="${RUN_NAME}_${max_seq_len}_lora"
    fi
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
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT
echo COUNT_NODE= $COUNT_NODE
echo RANK= $RANK
echo GPU=${GPU}
echo WORKERS=$WORKERS
echo "Running ${RUN_NAME}"


if [ $lora_enabled = true ]; then
    echo "lora is enabled"
    config_file="./accelerate_configs/accelerate_config_zero2.yaml"
    echo $config_file
else
    echo "lora is disabled"
    config_file="./accelerate_configs/accelerate_config_zero2.yaml"
    echo $config_file
fi

per_device_train_batch_size=1
gradient_accumulation_steps=$(($global_batch_size / ($per_device_train_batch_size * $GPU)))
echo gradient_accumulation_steps=$global_batch_size / \($per_device_train_batch_size \* $GPU\) = $gradient_accumulation_steps

accelerate launch --config_file=$config_file \
    --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines=${COUNT_NODE} --num_processes=${GPU} \
    train_qwen2_vl_vae.py --model_name_or_path $model_name_or_path \
    --data_config_file $DATA_CONFIG_FILE \
    --conv_template "plain" \
    --run_name $RUN_NAME \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --hub_model_id $hub_model_id \
    --hub_token "$hub_token" \
    --push_to_hub $push_to_hub \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers $WORKERS \
    --report_to wandb \
    --do_train \
    --lora_enabled $lora_enabled \
    --qlora_enabled $qlora_enabled \
    --max_seq_len $max_seq_len \
    --resume_from_checkpoint "$resume_from_checkpoint" \
    --min_pixels $min_pixels \
    --max_pixels $max_pixels \
    --use_liger_kernel $use_liger_kernel \
    --vae_class_name $vae_class_name \
    --vae_model_name_or_path $vae_model_name_or_path \
    --vae_subfolder $vae_subfolder \
    --vae_latent_channel_size $vae_latent_channel_size \
    --vae_temporal_compress_rate $vae_temporal_compress_rate \
    --vae_width_compress_rate $vae_width_compress_rate \
    --vae_height_compress_rate $vae_height_compress_rate \
    --post_vae_patch_size $post_vae_patch_size \
    --do_pretrain $do_pretrain
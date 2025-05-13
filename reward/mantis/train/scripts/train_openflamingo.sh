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
    DATA_CONFIG_FILE="./data_configs/mantis_instruct.yaml"  # change to this for offical training
fi
if [ "$TRANSFORMERS_OFFLINE" = 1 ]; then
    echo "Warning: Offline mode is enabled. Using local copy of models"
    echo "Please set the local_model_path in the script"
    exit 1 # comment this line after setting the local_model_path
    model_name_or_path="{local_model_path}"
else
    model_name_or_path="openflamingo/OpenFlamingo-9B-vitl-mpt7b"
    tokenizer_path="mosaicml/mpt-7b"
    lang_encoder_path="mosaicml/mpt-7b"
    clip_vision_encoder_path="ViT-L-14"
    clip_vision_encoder_pretrained="openai"
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

hf_hub_user_name="Mantis-VL" # set this will push the model to your hub after training
max_seq_len=2048 # openflamingo use cross attention for vision processing, which does not add tokens to the input sequence
lora_enabled=false
qlora_enabled=false
OUTPUT_DIR="../../checkpoints"
global_batch_size=128

RUN_NAME="mantis-9b-openflamingo"
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
    config_file="./accelerate_configs/accelerate_config_zero3.yaml"
    echo $config_file
fi

per_device_train_batch_size=1
gradient_accumulation_steps=$(($global_batch_size / ($per_device_train_batch_size * $GPU)))
echo gradient_accumulation_steps=$global_batch_size / \($per_device_train_batch_size \* $GPU\) = $gradient_accumulation_steps

accelerate launch --config_file=$config_file \
    --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --num_machines=${COUNT_NODE} --num_processes=${GPU} \
    train_openflamingo.py --model_name_or_path $model_name_or_path \
    --tokenizer_path $tokenizer_path \
    --lang_encoder_path $lang_encoder_path \
    --clip_vision_encoder_path $clip_vision_encoder_path \
    --clip_vision_encoder_pretrained $clip_vision_encoder_pretrained \
    --data_config_file $DATA_CONFIG_FILE \
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
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --dataloader_num_workers $WORKERS \
    --report_to wandb \
    --do_train \
    --lora_enabled $lora_enabled \
    --qlora_enabled $qlora_enabled \
    --max_seq_len $max_seq_len \
    --resume_from_checkpoint "$resume_from_checkpoint" \
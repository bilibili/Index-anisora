#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/dataset/fd5061f6/sat_pretrained/roberta

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=4
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
source $main_dir/config/model_roberta_large.sh
echo $MODEL_TYPE

task_name=$1

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

dataset_name="$task_name"
if [[ "$task_name" == "wsc" ]]; then
  dataset_name="wsc.fixed"
fi

en_data="hf://super_glue/${dataset_name}/train"
eval_data="hf://super_glue/${dataset_name}/validation"

config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-${dataset_name}-best-WOPT-\
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 8000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --distributed-backend nccl \
       --lr-decay-style linear \
       --checkpoint-activations \
       --fp16 \
       --eval-interval 100 \
       --save checkpoints/ \
       --split 1 \
       --eval-batch-size 32 \
       --valid-data ${eval_data} \
       --strict-eval \
       --warmup 0.1
"



gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"

((port=$RANDOM+10000))

run_cmd="${OPTIONS_NCCL} deepspeed --include=localhost:4,5,6,7 --master_port ${port} --hostfile ${HOST_FILE_PATH} finetune_roberta_${task_name}.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

#! /bin/bash

CHECKPOINT_PATH=$1
if [[ "$1" == "" ]] || [[ "$2" == "" ]];
then
    echo "Please pass in two root folders to save model and data!"
    exit
fi

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=6
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="clip"
MODEL_ARGS="--num-finetune-classes 10 \
            --image-size 384 384"

OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="$2/train"
eval_data="$2/test"


config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-cifar10 \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 1000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 6000 \
       --eval-interval 100 \
       --save "$CHECKPOINT_PATH/checkpoints" \
       --split 1 \
       --strict-eval \
       --eval-batch-size 8 \
       --zero-stage 1 \
       --lr 0.00002 \
       --batch-size 4 \
       --md_type $MODEL_TYPE
"


run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --include localhost:0,1 --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_clip_cifar10.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

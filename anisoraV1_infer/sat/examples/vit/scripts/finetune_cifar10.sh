#! /bin/bash

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="vit-base-patch16-224-in21k"
MODEL_ARGS="--num-finetune-classes 10 \
            --finetune-resolution 384 384"

# OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="./train"
eval_data="./test"


gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-cifar10 \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 1500 \
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
       --save "./checkpoints" \
       --split 1 \
       --strict-eval \
       --eval-batch-size 16 \
       --zero-stage 1 \
       --lr 0.00005 \
       --batch-size 16 \
       --from_pretrained $MODEL_TYPE
"

              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_vit_cifar10.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

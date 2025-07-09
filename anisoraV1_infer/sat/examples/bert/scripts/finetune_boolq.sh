#! /bin/bash

CHECKPOINT_PATH=$1
if [[ "$1" == "" ]] || [[ "$2" == "" ]];
then
    echo "Please pass in two root folders to save model and data!"
    exit
fi

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="bert-base-uncased"

OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=warning NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="hf://super_glue/boolq/train"
eval_data="hf://super_glue/boolq/validation"
test_data="hf://super_glue/boolq/test"

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-boolq \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 1000 \
       --resume-dataloader \
       --train-data ${en_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --fp16 \
       --save-interval 1000 \
       --eval-interval 100 \
       --save "$CHECKPOINT_PATH/checkpoints" \
       --split 1 \
       --strict-eval \
       --eval-batch-size 8 \
       --zero-stage 0 \
       --lr 0.00002 \
       --batch-size 8 \
       --data_root $2 \
       --md_type $MODEL_TYPE \
       --save-args
"


run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --include localhost:0,1,2,3,4,5,6,7 --hostfile ${HOST_FILE_PATH} finetune_bert_boolq.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

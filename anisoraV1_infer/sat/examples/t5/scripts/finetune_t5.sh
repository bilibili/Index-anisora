#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=2
MP_SIZE=1
source $1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

full_data="/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_4leveltokens/merge.bin"
small_data="/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_4leveltokens/zijian/zijian.bin.part_0.cogdata"

config_json="$main_dir/config/config_t5_large.json"
gpt_options=" \
       --experiment-name finetune-t5-test \
       --tokenizer-type Fake \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       $MODEL_ARGS \
       --train-iters 200000 \
       --resume-dataloader \
       --train-data ${small_data} \
       --split 1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --checkpoint-activations \
       --save-interval 5000 \
       --eval-interval 1000 \
       --save /root/checkpoints \
       --fp16
"
       # --load pretrained/cogview/cogview-base


gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"
              

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_t5.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x

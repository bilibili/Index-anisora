#!/bin/bash

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

CHECKPOINT_PATH=/dataset/fd5061f6/sat_pretrained/cogview/cogview-caption
NLAYERS=48
NHIDDEN=2560
NATT=40
MAXSEQLEN=1089
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=1.03
TOPK=200

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

gpt_options=" \
       --tokenizer-type cogview \
       --img-tokenizer-path /dataset/fd5061f6/sat_pretrained/vqvae/l1+ms-ssim+revd_percep.pt \
       --mode inference \
       --distributed-backend nccl \
       --max-sequence-length 1089 \
       --sandwich-ln \
       --fp16 \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --temperature $TEMP \
       --top_k $TOPK \
       --sandwich-ln \
       --input-source coco_select.txt \
       --output-path . \
       --batch-size 60 \
       --max-inference-batch-size 12 \
    "

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} inference_cogview_caps.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
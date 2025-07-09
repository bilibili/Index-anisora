#!/bin/bash
# SAT_HOME=/raid/dm/sat_models

NLAYERS=48
NHIDDEN=2560
NATT=40
MAXSEQLEN=1089
MPSIZE=1

#SAMPLING ARGS
TEMP=1.03
TOPK=200

# SAT_HOME=$SAT_HOME \
python inference_cogview.py \
       --tokenizer-type cogview \
       --mode inference \
       --distributed-backend nccl \
       --max-sequence-length 1089 \
       --sandwich-ln \
       --fp16 \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --num-attention-heads $NATT \
       --temperature $TEMP \
       --top_k $TOPK \
       --sandwich-ln \
       --input-source ./input.txt \
       --output-path samples_text2image \
       --batch-size 4 \
       --max-inference-batch-size 8 \
       $@



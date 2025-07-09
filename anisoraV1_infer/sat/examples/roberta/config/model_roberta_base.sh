MODEL_TYPE="roberta-base"
MODEL_ARGS="--num-layers 12 \
            --vocab-size 50265 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-sequence-length 514 \
            --hidden-dropout 0.1 \
            --attention-dropout 0.1 \
            --checkpoint-activations \
            --checkpoint-num-layers 1 \
            --load ${CHECKPOINT_PATH}/$MODEL_TYPE"
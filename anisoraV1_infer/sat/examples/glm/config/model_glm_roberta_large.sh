MODEL_TYPE="blocklm-roberta-large"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 513 \
            --tokenizer-model-type roberta \
            --tokenizer-type glm_GPT2BPETokenizer \
            --load ${CHECKPOINT_PATH}/glm-large-en-blank"
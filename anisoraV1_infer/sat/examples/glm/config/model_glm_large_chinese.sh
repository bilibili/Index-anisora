MODEL_TYPE="blocklm-large-chinese"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 1025 \
            --tokenizer-type glm_ChineseSPTokenizer \
            --tokenizer-model-type glm-large \
            --load ${CHECKPOINT_PATH}/glm-large-zh"
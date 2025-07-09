MODEL_TYPE="blocklm-10B-chinese"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 48 \
            --hidden-size 4096 \
            --num-attention-heads 64 \
            --max-sequence-length 1025 \
            --tokenizer-type glm_ChineseSPTokenizer \
            --tokenizer-model-type glm-10b \
            --load ${CHECKPOINT_PATH}/glm-10b-zh"
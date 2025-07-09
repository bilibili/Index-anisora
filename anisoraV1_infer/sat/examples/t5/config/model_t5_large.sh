MODEL_TYPE="t5-large"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --vocab-size 32128 \
            --num-layers 24 \
            --hidden-size 1024 \
            --inner-hidden-size 4096 \
            --num-attention-heads 16 \
            --hidden-size-per-attention-head 64 \
            --max-sequence-length 513 \
            --relative-attention-num-buckets 32 \
            --layernorm-epsilon 1e-6 \
            --tokenizer-type hf_T5Tokenizer \
            --tokenizer-model-type t5-large"
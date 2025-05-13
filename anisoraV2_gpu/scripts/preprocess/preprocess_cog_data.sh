# export WANDB_MODE="offline"#
######test-t2v-5B-wukong
export HF_ENDPOINT="https://hf-mirror.com"
GPU_NUM=4 # 2,4,8
MODEL_PATH="/DATA/bvac/personal/opensora/zhipu/pretrained/hf2b/cache/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad"
MODEL_TYPE="cog"
DATA_MERGE_PATH="/DATA/bvac/personal/fastvideo/FastVideo-main/data/Image-Vid-Finetune-Src/merge.txt"
OUTPUT_DIR="/DATA/bvac/personal/fastvideo/FastVideo-main/data/Image-Vid-Finetune-cog480P-49f8fps"
VALIDATION_PATH="/DATA/bvac/personal/fastvideo/FastVideo-main/assets/prompt1.txt"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_vae_latents.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=720 \
    --num_frames=49 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 12

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR 

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt_txt $VALIDATION_PATH
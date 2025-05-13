export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#based on fm1500
torchrun --nnodes 1 --nproc_per_node 8  \
    fastvideo/distill_i2v_cog.py  \
    --seed 42 \
    --pretrained_model_name_or_path /DATA/bvac/personal/opensora/zhipu/pretrained/danzhen-i2v-97f-720P-16fps/danzhen-i2v-97f-720P-16fps  \
    --dit_model_name_or_path /DATA/bvac/personal/opensora/zhipu/pretrained/danzhen-i2v-97f-720P-16fps/danzhen-i2v-97f-720P-16fps  \
    --model_type "cog"  \
    --cache_dir .cache  \
    --data_json_path /DATA/bvac/personal/fastvideo/FastVideo-main/data/Image-Vid-Finetune-cog480P-49f12fps-fixvae/videos2caption.json  \
    --validation_prompt_dir /DATA/bvac/personal/fastvideo/FastVideo-main/data/Image-Vid-Finetune-cog480P-49f12fps-fixvae/validation  \
    --gradient_checkpointing  \
    --train_batch_size=1  \
    --num_latent_t 24  \
    --sp_size 1  \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=20000 \
    --learning_rate=3e-6 \
    --mixed_precision="bf16" \
    --checkpointing_steps=400 \
    --validation_steps 15000000 \
    --validation_sampling_steps "2,4,8"  \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir=outputs-test-cog_i2v_5B-wukong49-distill-3e-6-fixvae2 \
    --tracker_project_name video3w480Ptest_cog49i2v_fix_distill-3e-6-fixvae2  \
    --num_frames  49  \
    --shift 17  \
    --validation_guidance_scale "1.0"  \
    --num_euler_timesteps 50  \
    --multi_phased_distill_schedule "4000-1"  \
    --not_apply_cfg_solver 
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


torchrun --nnodes 5 --nproc_per_node 8  --node_rank=${NODE_RANK} --master_addr=10.156.32.11 --master_port=14431 \
    fastvideo/distill_i2v_cog720-4133-nonegrid.py  \
    --seed 42 \
    --pretrained_model_name_or_path /DATA/bvac/personal/opensora/zhipu/pretrained/mix26  \
    --dit_model_name_or_path /DATA/bvac/personal/opensora/zhipu/pretrained/mix26  \
    --model_type "cog"  \
    --cache_dir /data/docker/data/video14w/.cache \
    --data_json_path2 /DATA/bvac/personal/fastvideo/make-data/all33-41-49/videos2caption41.json \
    --data_json_path3 /DATA/bvac/personal/fastvideo/make-data/all33-41-49/videos2caption33.json \
    --validation_prompt_dir /data/docker/data/video14w/outputs/validation \
    --gradient_checkpointing  \
    --train_batch_size=1  \
    --num_latent_t 24  \
    --sp_size 1  \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=20000 \
    --learning_rate=1.5e-6 \
    --mixed_precision="bf16" \
    --checkpointing_steps=64 \
    --validation_steps 15000000 \
    --validation_sampling_steps "2,4,8"  \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir=outputs-cog_i2sv_5B_down_vae_eryu4133_5gpus_fix_text_rot_fvfm_mix26_distill1d5e6 \
    --tracker_project_name cog_i2v_5B_down_vae_eryu4133_5gpus_fix_text_rot_fvfm_mix26_distill1d5e6  \
    --num_frames  41  \
    --shift 17  \
    --validation_guidance_scale "1.0"  \
    --num_euler_timesteps 50  \
    --multi_phased_distill_schedule "4000-1"  \
    --not_apply_cfg_solver 
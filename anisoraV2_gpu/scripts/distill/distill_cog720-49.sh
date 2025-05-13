export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#based on fm1500
torchrun --nnodes 3 --nproc_per_node 8  --node_rank=${NODE_RANK} --master_addr=10.156.32.11 --master_port=14431 \
    fastvideo/distill_i2v_cog720-49.py  \
    --seed 42 \
    --pretrained_model_name_or_path /DATA/bvac/personal/opensora/zhipu/pretrained/danzhen-i2v-97f-720P-16fps/danzhen-i2v-97f-720P-16fps-1600steps_bs24-fm6000  \
    --dit_model_name_or_path /DATA/bvac/personal/opensora/zhipu/pretrained/danzhen-i2v-97f-720P-16fps/danzhen-i2v-97f-720P-16fps-1600steps_bs24-fm6000  \
    --model_type "cog"  \
    --cache_dir /data/docker/data/video14w/.cache \
    --data_json_path /data/docker/data/video14w/outputs/videos2caption.json \
    --validation_prompt_dir /data/docker/data/video14w/outputs/validation \
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
    --checkpointing_steps=250 \
    --validation_steps 15000000 \
    --validation_sampling_steps "2,4,8"  \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --log_validation \
    --output_dir=outputs-cog_i2v_5B_down_vae_eryu49_3gpus_fix_text_rot_fvfm6k_distill \
    --tracker_project_name cog_i2v_5B_down_vae_eryu49_3gpus_fix_text_rot_fvfm6k_distill  \
    --num_frames  49  \
    --shift 17  \
    --validation_guidance_scale "1.0"  \
    --num_euler_timesteps 50  \
    --multi_phased_distill_schedule "4000-1"  \
    --not_apply_cfg_solver 
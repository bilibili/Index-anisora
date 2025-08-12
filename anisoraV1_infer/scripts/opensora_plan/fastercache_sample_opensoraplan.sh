output='results/'
torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/fastercache_sample_opensoraplan.py --config configs/opensora_plan/fastercache_sample.yaml --save_img_path ${output}/
output='results/'
torchrun --standalone --nproc_per_node=1 scripts/latte/fastercache_sample_latte.py --config configs/latte/fastercache_sample.yaml --save_img_path ${output}


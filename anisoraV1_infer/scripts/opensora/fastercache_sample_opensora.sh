output='results/'
torchrun --standalone --nproc_per_node=1 scripts/opensora/fastercache_sample_opensora.py --config configs/opensora/fastercache_sample.yaml --save-dir ${output}/
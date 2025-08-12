output='results/'
node=2
pname=your_partition_name
srun -p ${pname} --gres=gpu:${node} --quotatype=reseverd --ntasks-per-node=${node} -n${node} -N1 --cpus-per-task=12 python scripts/latte/fastercache_sample_multi_device_latte.py --config configs/latte/fastercache_sample.yaml --save_img_path ${output} 
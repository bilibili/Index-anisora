
output='results/'
node=2
pname=your_partition_name
srun -p ${pname} --gres=gpu:${node} --quotatype=reseverd --ntasks-per-node=${node} -n${node} -N1 --cpus-per-task=12 python scripts/opensora_plan/fastercache_sample_multi_device_opensoraplan.py --config configs/opensora_plan/fastercache_sample_221.yaml --save_img_path ${output}/
#!/bin/bash

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
cd $SCRIPT_ROOT

./client job run --job_name=npu_cog15B_0 \
--worker_hardware=CPU:128,MEM:1800,GPU:16,DISK:2000 \
--namespace=ai-npu910 \
--cluster=jscs04 \
--worker_num=1 \
--image_url=hub.bilibili.co/nyx-base/910b:ttv_dev \
--entrypoint_type=bash \
--entrypoint=/root/sleep.sh \
--framework=tch \
--task_group=ai.llm \
-E e_enable_ssh=true \
-E e_instance_retain=true \
-E e_enable_webshell=true \
-E e_enable_notebook=true \
-E e_host_ipc=true \
-E e_network_mode=host \
-E e_run_timeout_minute=0 \
-E NCCL_DEBUG=info \
-E NCCL_IB_GID_INDEX=3 \
-E NCCL_IB_HCA="mlx5_bond_0,mlx5_bond_1" \
-E NCCL_NET_GDR_LEVEL=PHB \
-E CUDA_DEVICE_MAX_CONNECTIONS=1 \
-E e_need_fuse=true \
-E e_alluxio_fuse_memory=32 \
-E e_hdfs_dataset="/department/ai/llm_dataset:/mnt/hdfs"

#!/bin/bash
export COMBINED_ENABLE=1

# set alluxio envs
APPNAME=alluxio
export ALLUXIO_HOME=/data/service/${APPNAME}
export ALLUXIO_CONF_DIR=${ALLUXIO_HOME}/conf-fuse
export ALLUXIO_LOGS_DIR=/data/log/${APPNAME}
export PATH=${ALLUXIO_HOME}/bin:$PATH

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"

#更新代码到安装目录
cd swissarmytransformer-npu_t_sp
pip install -e . 


cd $SCRIPT_ROOT/../sat
torchrun --nproc_per_node=${PET_NPROC_PER_NODE} --nnodes=${PET_NNODES} --node_rank=${PET_NODE_RANK} --master_addr=${PET_MASTER_ADDR} --master_port=${PET_MASTER_PORT}  train_video.py  --base configs/cogvideox_5b_sft_fix_pretrain_gpu_720.yaml --seed 1234

#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ACCELERATOR=npu
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export COMBINED_ENABLE=1

# set alluxio envs
APPNAME=alluxio
export ALLUXIO_HOME=/data/service/${APPNAME}
export ALLUXIO_CONF_DIR=${ALLUXIO_HOME}/conf-fuse
export ALLUXIO_LOGS_DIR=/data/log/${APPNAME}
export PATH=${ALLUXIO_HOME}/bin:$PATH

# pip install 
pip install ./env_file/torch-2.1.0+cpu-cp310-cp310-linux_x86_64.whl
pip install torch-npu==2.1.0.post3 
pip install torchvision==0.16.0 
pip install boto3
SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

cd swissarmytransformer-npu_t_sp
pip install -e . 

cd $SCRIPT_ROOT/../sat

torchrun --nproc_per_node=${PET_NPROC_PER_NODE} --nnodes=${PET_NNODES} --node_rank=${PET_NODE_RANK} --master_addr=${PET_MASTER_ADDR} --master_port=${PET_MASTER_PORT}  train_video.py  --base configs/cogvideox_5b_sft_fix_pretrain_npu_720.yaml --seed 1234

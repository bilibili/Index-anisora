MODEL_TYPE="clip"
if [ ! -d "${CHECKPOINT_PATH}/$MODEL_TYPE" ]
then
    if [ ! -f "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" ]
    then
        wget "https://cloud.tsinghua.edu.cn/f/bd29f0537f9949e6a4fb/?dl=1" -O "${CHECKPOINT_PATH}/$MODEL_TYPE.zip"
    fi
    unzip "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" -d "${CHECKPOINT_PATH}"
fi
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE \
            --num-finetune-classes 10 \
            --image-size 224 224"

if [ -d "${CHECKPOINT_PATH}/clip-vit-base-patch32" ]
then
    PRETRAIN_PATH="${CHECKPOINT_PATH}"
else
    PRETRAIN_PATH=""
fi
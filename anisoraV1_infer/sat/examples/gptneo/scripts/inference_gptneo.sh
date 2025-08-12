#! /bin/bash

CHECKPOINT_PATH=$1
if [[ "$1" == "" ]];
then
    echo "Please pass in root folder to save model!"
    exit
fi

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"

run_cmd="${OPTIONS_SAT} python inference_gptneo.py --mode inference"
echo ${run_cmd}
eval ${run_cmd}

set +x

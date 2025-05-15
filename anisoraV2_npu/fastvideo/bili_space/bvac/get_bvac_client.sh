#!/bin/bash
set -e

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
cd $SCRIPT_ROOT

rm -rf client-linux86-64
wget https://ml.bilibili.co/api/download/client-linux86-64
chmod +x client-linux86-64
./client-linux86-64 update
./client config init

echo -e "\e[1;43;32mPls write \"user_token\" into \"bvac_cmd/.conf/config.yaml\" \e[0m"
echo -e "\e[1;43;32mPls write \"user_token\" into \"bvac_cmd/.conf/config.yaml\" \e[0m"
echo -e "\e[1;43;32mPls write \"user_token\" into \"bvac_cmd/.conf/config.yaml\" \e[0m"

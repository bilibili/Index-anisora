import logging
import os
import re
import sys

from py_common import BossClient

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from config import g_config

model_dir = os.path.join(current_dir, "../models")
os.makedirs(model_dir, exist_ok=True)
model_list = [
    ('cv_platform','panxiaoan/siqi/cogdata_index/ckpts/sat_ckpts/Cogvideo_caption/mp_rank_00_model_states.pt', "models/CogvideoX/ckpts/1000"),#yinmingyu/tmp/1203/mp_rank_00_model_states.pt
    ('danbooru2021','cogvideo-pretrained/3d-vae.pt', "models/CogvideoX/ckpts"),
    ('danbooru2021','cogvideo-pretrained/t5-v1_1-xxl-fixed', "models/CogvideoX/ckpts"),
]


def prepare_models():
    logging.info("prepare models...")
    print("prepare models...")
    dirs = model_list
    current_bucket = None
    boss_client = None
    for bucket, dirname,local_dir in dirs:
        if current_bucket != bucket:
            boss_client = BossClient(g_config[bucket])
            current_bucket = bucket
        download_dir(boss_client, dirname, local_dir)
    with open(os.path.join(model_dir, "CogvideoX/ckpts", "latest"), "w") as f:
        f.write("1000")
    logging.info("download model succ")
    print("download model succ")

def download_dir(boss_client, boss_path, local_dir):
    files = boss_client.list_objects(boss_path, basename=False)
    dir_name = boss_path.split('/')[-1]
    # print(files)
    for file in files:
        if file.endswith('/'):
            continue
        local_path = re.sub(boss_path, f"{local_dir}/{dir_name}", file)
        if not os.path.isfile(local_path):
            print(local_path)
            boss_client.download_file(file, filename=local_path)

if __name__ == "__main__":
    prepare_models()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:38:35 2024

@author: yinmingyu
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:54:18 2022

@author: yinmingyu
"""

import time
import os
import hashlib
import requests
import json
from argparse import ArgumentParser
import boto3
from botocore.config import Config


class Boss:
    def __init__(self):
        my_config = Config(s3={'addressing_style': 'path'})  # 必须: 必须添加这项配置

        self.s3_client = boto3.client('s3',
                                      aws_access_key_id='<REDACTED_SECRET>',  # 替换为申请 Bucket 时提供的 AccessKey
                                      aws_secret_access_key='<REDACTED_SECRET>',  # 替换为申请 Bucket 时提供的 SecretKey
                                      region_name='jssz',  # 替换为申请 Bucket 时提供的 Region
                                      endpoint_url='http://jssz-boss.bilibili.co',  # 替换为申请 Bucket 时提供的 Endpoint
                                      config=my_config)

        self.s3_resource = boto3.resource('s3',
                                          aws_access_key_id='<REDACTED_SECRET>',  # 替换为申请 Bucket 时提供的 AccessKey
                                          aws_secret_access_key='<REDACTED_SECRET>',  # 替换为申请 Bucket 时提供的 SecretKey
                                          region_name='jssz',  # 替换为申请 Bucket 时提供的 Region
                                          endpoint_url='http://jssz-boss.bilibili.co',  # 替换为申请 Bucket 时提供的 Endpoint
                                          config=my_config)
        self.BucketName = "cv_platform"


appkey = "<REDACTED_SECRET>"
secret = "<REDACTED_SECRET>"

if __name__ == '__main__':
    boss = Boss()
    for i in range(1):
        # boss_url_fg = boss.s3_client.generate_presigned_url('get_object', Params={'Bucket': boss.BucketName, 'Key': 'yinmingyu/test/Nor_Fg_tha_sdv9_fgv3_rename/{}.png'.format(i)}, ExpiresIn=3600)
        boss_url_fg = boss.s3_client.generate_presigned_url('get_object', Params={'Bucket': boss.BucketName, 'Key': 'seasonyang/data/vae_feature_fps16_720p/video_clips/bili_ogv_clip/part-00010/254751358_570501069-Scene-013.pt'}, ExpiresIn=3600)
        print(boss_url_fg)

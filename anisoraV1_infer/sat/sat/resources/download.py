# -*- encoding: utf-8 -*-
'''
@File    :   download.py
@Time    :   2022/06/05 17:08:29
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import requests
import threading
from tqdm import tqdm
from filelock import FileLock
from .urls import MODEL_URLS

def download_with_progress_bar(save_path, url, chunk_size=2048):
    resume_header = None
    file_size_downloaded = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        file_size_downloaded = os.path.getsize(save_path)
        resume_header = {'Range': f'bytes={file_size_downloaded}-'}

    response = requests.get(url, stream=True, headers=resume_header)
    total_size = int(response.headers.get('content-length', 0)) + file_size_downloaded
    if total_size == file_size_downloaded:
        return
    
    with open(save_path, 'ab') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path, initial=file_size_downloaded) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

def auto_create(name, *, path=None, url=None):
    if path is None:
        path = os.getenv('SAT_HOME', '~/.sat_models')
    path = os.path.expanduser(path)
    model_path = os.path.join(path, name)
    if url == 'local':
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    lock = FileLock(model_path + '.lock', mode=0o777)
    with lock:
        if url is None:
            url = MODEL_URLS[name]
        if os.path.isdir(model_path) and not url.startswith('r2://'):
            pass
        elif os.path.isdir(model_path) and url.startswith('r2://') and url.endswith('.zip'):
            pass
        else:
            print(f'Downloading models {url} into {path} ...')
            try:
                if url.startswith('r2://'):
                    download_s3(path, url[5:])
                else:
                    file_path = os.path.join(path, name + '.zip')
                    download_with_progress_bar(file_path, url)
            except Exception as e:
                print(f'Failed to download or check, if you already had the zip file, please unzip it manually as {model_path}!')
                raise e
        # unzip
        if not os.path.isdir(model_path):
            import zipfile
            file_path = os.path.join(path, name + '.zip')
            print(f'Unzipping {file_path}...')
            f = zipfile.ZipFile(file_path, 'r')
            f.extractall(path=path)
            assert os.path.isdir(model_path), f'Unzip failed, or the first-level folder in zip is not {name}.'
    return model_path # must return outside the `with lock` block

SAT_ACCOUNT = 'c8a00746a80e06c4632028e37de24d6e'
SAT_ACCESS_KEY = 'eb4d69e273848089c7f9b9599cdcd983'
SAT_SECRET_KEY = '367e9b21fef313f187026320016962b47b74ca4ada7d64d551c43c51e195d7a5'
SAT_BUCKET = 'sat'

def download_s3(local_dir, remote_uri):
    '''Download remote_dir into (under) local_dir
    '''
    import boto3
    s3_resource = boto3.resource('s3',
        endpoint_url = f'https://{SAT_ACCOUNT}.r2.cloudflarestorage.com',
        aws_access_key_id = f'{SAT_ACCESS_KEY}',
        aws_secret_access_key = f'{SAT_SECRET_KEY}'
        )
    client = boto3.client('s3',
        endpoint_url = f'https://{SAT_ACCOUNT}.r2.cloudflarestorage.com',
        aws_access_key_id = f'{SAT_ACCESS_KEY}',
        aws_secret_access_key = f'{SAT_SECRET_KEY}',
        verify=False
        )
    bucket = s3_resource.Bucket(SAT_BUCKET) 
    transfer_config = boto3.s3.transfer.TransferConfig(
        use_threads=True,
        multipart_threshold=8*1024*1024,
        max_concurrency=64,
        multipart_chunksize=8*1024*1024,
    )
    # remote_uri is file
    if '.' in os.path.basename(remote_uri):
        bucket.download_file(remote_uri, os.path.join(local_dir, os.path.basename(remote_uri)),Callback=ProgressPercentage(client, SAT_BUCKET, remote_uri), Config=transfer_config)
        return 
    # uri is path
    remote_dir = remote_uri
    key_prefix = remote_dir.split('/')[:-1]
    for obj in bucket.objects.filter(Prefix = remote_dir):
        key_suffix = obj.key[len(key_prefix):] # remote_dir/xxx/xxx.zip
        target_dir = os.path.join(local_dir, os.path.dirname(key_suffix))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        # skip if exists
        if os.path.exists(os.path.join(local_dir, key_suffix)) and os.path.getsize(os.path.join(local_dir, key_suffix)) == obj.size:
            continue
        bucket.download_file(obj.key, os.path.join(local_dir, key_suffix),Callback=ProgressPercentage(client, SAT_BUCKET, obj.key), Config=transfer_config) 


class ProgressPercentage(object):
    ''' Progress Class
    Class for calculating and displaying download progress
    '''
    def __init__(self, client, bucket, filename):
        ''' Initialize
        initialize with: file name, file size and lock.
        Set seen_so_far to 0. Set progress bar length
        '''
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount):
        ''' Call
        When called, increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size 
        and prints progress bar.
        '''
        # To simplify we'll assume this is hooked up to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round((float(self._seen_so_far) / float(self._size)) * (self.prog_bar_len - 6), 1)
            current_length = int(round(ratio))

            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

            bars = '+' * current_length
            output = bars + ' ' * (self.prog_bar_len - current_length - len(str(percentage)) - 1) + str(percentage) + '% ' + self.convert_bytes(self._seen_so_far) + ' / ' + self.convert_bytes(self._size) + ' ' * 5

            if self._seen_so_far != self._size:
                sys.stdout.write(output + '\r')
            else:
                sys.stdout.write(output + '\n')
            sys.stdout.flush()

    def convert_bytes(self, num):
        ''' Convert Bytes
        Converts bytes to scaled format (e.g KB, MB, etc.)
        '''
        step_unit = 1000.0
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if num < step_unit:
                return "%3.1f %s" % (num, x)
            num /= step_unit
        

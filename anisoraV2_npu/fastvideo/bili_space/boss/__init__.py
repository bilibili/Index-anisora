# -*- coding: utf-8 -*-

import logging
import mimetypes
import os
import time

import boto3
from botocore.config import Config
from botocore import exceptions


class BossClient:
    """
    BOSS对象存储访问客户端
    配置文件包含endpoint, bucket, access_key, secret_key,
    同时需要提供object_path作为要访问的相对路径。
    """

    def __init__(self, conf):
        clientConfig = {
            "signature_version": "s3v4",
            "s3": {
                "addressing_style": "path"
            },
            "retries": {
                "max_attempts": 0
            }
        }
        if "config" in conf:
            clientConfig.update(conf["config"])

        self._bucket = conf["bucket"]
        self._object_path = conf.get("object_path", "")
        self._s3_client = boto3.client(
            's3',
            use_ssl=False,
            verify=False,
            endpoint_url=conf["endpoint"],
            aws_access_key_id=conf["access_key"],
            aws_secret_access_key=conf["secret_key"],
            config=Config(**clientConfig))

    def metric_wrapper(func):

        def wrapper(self, *args, **kw):
            code, tic = 0, time.time()
            try:
                return func(self, *args, **kw)
            except exceptions.ClientError as e:
                code = e.response['Error']['Code']
                if func.__name__ == "exist_object" and code == "404":
                    # The key does not exist.
                    return False
                elif code == "NoSuchKey":
                    logging.warning(f"s3 Client NoSuchKey: {e}")
                    raise e
                else:
                    # Something else has gone wrong.
                    logging.error(f"s3 ClientError: {e}")
                    raise e
            except Exception as e:
                logging.error(f"s3 exception: {e}")
                raise e

        return wrapper

    def _get_object_abs_path(self, object_name):
        return os.path.join(self._object_path, object_name)

    def create_presigned_url(self, object_name, expiration=86400):
        """
        生成带签名的url
        @param object_name: 对象相对路径
        @param expiration: 链接过期时间（秒），默认一天
        """
        object_name = self._get_object_abs_path(object_name)
        try:
            response = self._s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self._bucket,
                    'Key': object_name
                },
                ExpiresIn=expiration)
        except Exception as e:
            logging.error(e)
            return None
        return response

    @metric_wrapper
    def delete_object(self, object_name):
        """
        从远程删除对象
        @param object_name: 远程对象相对路径
        """
        object_name = self._get_object_abs_path(object_name)
        self._s3_client.delete_object(Bucket=self._bucket, Key=object_name)

    @metric_wrapper
    def download_file(self, object_name, filename=None):
        """
        下载远程文件到本地
        @param object_name: 远程文件相对路径
        @param filename: 本地文件路径
        """
        object_name = self._get_object_abs_path(object_name)
        if filename is None:
            filename = os.path.basename(object_name)
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        self._s3_client.download_file(self._bucket, object_name, filename)

    @metric_wrapper
    def get_object(self, object_name):
        """
        从BOSS获取远程对象字节到本地内存
        @param object_name: 远程对象相对路径
        @return 成功返回对象字节数据，否则抛异常
        """
        object_name = self._get_object_abs_path(object_name)
        rsp = self._s3_client.get_object(Bucket=self._bucket, Key=object_name)
        return rsp['Body'].read()

    @metric_wrapper
    def get_object_size(self, object_name):
        """
        获取远程对象大小（字节）
        @param object_name: 远程对象相对路径
        @return 成功返回对象字节数，否则抛异常
        """
        object_name = self._get_object_abs_path(object_name)
        response = self._s3_client.head_object(
            Bucket=self._bucket,
            Key=object_name,
        )
        if response.get('ContentLength'):
            return int(response['ContentLength'])
        return -1

    @metric_wrapper
    def get_object_last_modified(self, object_name):
        """
        获取远程对象最近修改时间 (utc datetime)
        @param object_name: 远程对象相对路径
        @return 成功返回时间，否则抛异常
        """
        object_name = self._get_object_abs_path(object_name)
        response = self._s3_client.head_object(
            Bucket=self._bucket,
            Key=object_name,
        )
        if response.get('LastModified'):
            return response['LastModified']

    @metric_wrapper
    def list_objects(self, object_path=None, basename=True):
        """
        列举远程目录所有文件
        @param basename: 使用文件名代替绝对路径
        @param object_path: 远程相对路径
        @return 返回远程文件集合，失败抛异常
        """
        if object_path is None:
            object_path = self._object_path
        else:
            object_path = self._get_object_abs_path(object_path)
        all_keys = set()

        response = self._s3_client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=object_path,
        )

        for key in response.get('Contents', []):
            name = os.path.basename(key['Key']) if basename else os.path.relpath(key['Key'], self._object_path)
            all_keys.add(name)
        while response.get('IsTruncated', False):
            continuation_key = response['NextContinuationToken']
            response = self._s3_client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=object_path,
                ContinuationToken=continuation_key)
            for key in response['Contents']:
                name = os.path.basename(key['Key']) if basename else os.path.relpath(key['Key'], self._object_path)
                all_keys.add(name)

        return all_keys

    @metric_wrapper
    def put_object(self, obj_bytes, object_name):
        """
        将内存对象上传到BOSS
        @param obj_bytes: 位于内存的对象字节
        @param object_name: 远程对象相对路径
        """
        object_name = self._get_object_abs_path(object_name)
        self._s3_client.put_object(Body=obj_bytes,
                                   Bucket=self._bucket,
                                   Key=object_name)

    @metric_wrapper
    def upload_file(self, filename, object_name=None):
        """
        上传本地文件到BOSS
        @param filename: 本地文件路径
        @param object_name: 远程对象相对路径，默认使用本地文件名称
        """
        if object_name is None:
            object_name = os.path.basename(filename)
        object_name = self._get_object_abs_path(object_name)

        content_type = mimetypes.guess_type(
            filename)[0] or 'application/octet-stream'
        self._s3_client.upload_file(filename,
                                    self._bucket,
                                    object_name,
                                    ExtraArgs={"ContentType": content_type})

    @metric_wrapper
    def exist_object(self, object_name):
        """
        判断远程对象是否存在
        @param object_name: 远程对象相对路径
        @return 存在返回True，不存在返回False，异常则抛出
        """
        object_name = self._get_object_abs_path(object_name)
        self._s3_client.head_object(Bucket=self._bucket, Key=object_name)
        return True

    def upload_folder(self, folder_path):
        """
        上传文件夹到 S3
        :param folder_path: 本地文件夹绝对路径
        """
        assert os.path.isdir(folder_path), "Not directory!"
        # 遍历文件夹中的文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 获取文件的本地路径
                local_file_path = os.path.join(root, file)
                # 创建相对路径
                relative_path = os.path.relpath(local_file_path, os.path.dirname(folder_path))
                # 上传文件
                self.upload_file(local_file_path, relative_path)
        return

    @classmethod
    def get_client(cls):
        cfg = {}
        cfg["endpoint"] = os.environ.get("BOSS_CKPT_ENDPOINT", None)
        cfg["access_key"] = os.environ.get("BOSS_CKPT_ACCESS", None)
        cfg["secret_key"] = os.environ.get("BOSS_CKPT_SECRET", None)
        cfg["bucket"] = os.environ.get("BOSS_CKPT_BUCKET", None)
        cfg["object_path"] = os.environ.get("BOSS_CKPT_PATH", None)

        handle = None
        if not any(value is None for value in cfg.values()):
            handle = cls(conf=cfg)
        return handle

if "__main__" == __name__:
    handle = BossClient.get_client()
    if handle:
        handle.upload_folder("/workspace/fastvideo/assets")
        tmp = handle.list_objects(basename=False, object_path="assets")
        for i in tmp:
            handle.delete_object(i)
        pass

    print("end")

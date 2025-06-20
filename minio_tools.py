import os
import shutil
from minio import Minio
from minio.error import S3Error
from config import MINIO_SERVER_PORT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE

minio_client = Minio(
    MINIO_SERVER_PORT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

BUCKET_NAME = "ai-data"

def download_minio_folder(bucket_name, folder_name, local_path):
    try:
        # 列出文件夹中的所有对象
        objects = minio_client.list_objects(bucket_name, prefix=folder_name, recursive=True)
        
        num = 0
        for obj in objects:
            num += 1
            local_file_path = os.path.join(local_path, obj.object_name.lstrip('/'))
            print(f"local_file_path: {local_file_path}")

            # 下载文件
            try:
                minio_client.fget_object(bucket_name, obj.object_name, str(local_file_path))
            except S3Error as e:
                print(f"下载文件 {obj.object_name} 时出错: {e}")
                return False

        return True
    except S3Error as e:
        # 捕获 list_objects 可能抛出的异常
        print(f"MinIO 操作出错: {e}")
        return False
    except Exception as e:
        # 捕获所有其他异常，并只返回字符串
        print(f"未知错误: {str(e)}")
        return False

def delete_temp_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        return True
    return False


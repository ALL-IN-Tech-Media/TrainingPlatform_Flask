import os
from config import BUCKET_NAME, LOCAL_DATASET_DIR
from ultralytics import YOLO
from tools import validate_and_update_yaml_fields, convert_xml_to_txt
from minio_tools import download_minio_folder
from dataset_action import update_database_status
import ray
from yolov5.train import really_training



@ray.remote(max_calls=1, num_gpus=0.2)
def training(training_id, minio_folder_name, coco_path, pre_model_path, images_path, labels_path, epochs, img_size, batch_size):

    ## 1、首先判断一下该数据集是否已经下载
    if os.path.exists(images_path) and os.path.isdir(images_path):
        print(f"数据集已存在。")
    else:
        print(f"数据集不存在，开始下载。")
        if not os.path.exists(LOCAL_DATASET_DIR):
            os.makedirs(LOCAL_DATASET_DIR)
        is_load = download_minio_folder(BUCKET_NAME, minio_folder_name, LOCAL_DATASET_DIR)
        if not is_load:
            update_database_status(training_id, "初始化异常")
            return False

    ## 2、更新yaml文件，将其中的路径替换为实际训练路径
    train_path = os.path.join(images_path, "train")
    val_path = os.path.join(images_path, "val")
    is_val, categories = validate_and_update_yaml_fields(coco_path, train_path, val_path) # 从这里顺便获取类别
    if not is_val:
        update_database_status(training_id, "初始化异常")
        return False

    ## 3、检查数据集和图片是否完整（将xml文件转为txt文件）
    is_convert = convert_xml_to_txt(categories, labels_path)
    if not is_convert:
        update_database_status(training_id, "初始化异常")
        return False

    ## 4、开始训练
    update_database_status(training_id, "训练中")
    result = really_training(training_id, pre_model_path, coco_path, epochs, img_size, batch_size)
    update_database_status(training_id, "训练完成")
    
    return result



if __name__ == "__main__":
    pass

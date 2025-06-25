from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import ray
import os
import zipfile
import shutil
from config import BUCKET_NAME, LOCAL_DATASET_DIR, LOCAL_PRE_MODEL_DIR
from minio_tools import delete_temp_dir, download_minio_folder
from concurrent.futures import ThreadPoolExecutor
from train_api import fine_tuning_lora
from tools import validate_and_update_yaml_fields
from test import main_test
from ray_manage import RayTaskManage
ray.init(ignore_reinit_error=True)  # 初始化Ray

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})  # 允许所有来源的请求并支持凭证

executor = ThreadPoolExecutor(max_workers=10)
ray_manager = RayTaskManage()

# 设置图片文件夹的路径
IMAGE_FOLDER = '/Users/liuzizhen/Projects/AIData/'  # 替换为您的图片文件夹路径
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/train_fine_tuning', methods=['POST'])
def train_fine_tuning():
    data = request.get_json()
    training_id = data.get('training_id')
    user_id = data.get('user_id')
    dataset_id = data.get('dataset_id')
    training_type = data.get('training_type')
    model_name = data.get('model_name')
    epochs = data.get('epochs')
    batch_size = data.get('batch_size')
    max_length = data.get('max_length')
    save_epoch = data.get('save_epoch') # 每n轮保存一次模型
    gpu = data.get('gpu')
    save_dir = os.path.join(os.path.join(os.path.join(LOCAL_DATASET_DIR, str(user_id)), str(dataset_id)), 'save_models')
    if training_type == 'total_fine_tuning':
        print("total_fine_tuning")
    elif training_type == 'partial_fine_tuning':
        print("partial_fine_tuning")
    else:
        return jsonify({'code': '500', 'message': '任务类型错误', 'data':{}})
    
    ## 判断数据集是否存在
    dataset_path = os.path.join(os.path.join(os.path.join(LOCAL_DATASET_DIR, str(user_id)), str(dataset_id)), 'datasets', "data.json")
    if not os.path.exists(dataset_path):
        return jsonify({'code': '500', 'message': '数据集不存在', 'data':{}})
    
    ## 判断预训练模型是否存在
    model_path = os.path.join(LOCAL_PRE_MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'code': '500', 'message': '模型不存在', 'data':{}})
    
    # 创建ray任务去训练
    future = fine_tuning_lora.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    ray_manager.submit_training_task(training_id, future)
    ray_manager.tasks[training_id] = future
    return jsonify({'code': '200', 'message': '上传成功', 'data':{}})



@app.route('/train_detect', methods=['POST'])
def train_model():
    ## 1、从请求中获取参数
    data = request.get_json()
    training_id = data.get('id')
    minio_folder_name = data.get('user') + '/' + data.get('datasetName') + '/'
    task_type = data.get('taskType')
    epochs = int(data.get('epoch'))
    batch_size = int(data.get('batchSize'))
    img_size = int(data.get('imageSize'))
    images_path = ''
    labels_path = ''
    coco_path = ''
    pre_model_path = os.path.join(LOCAL_PRE_MODEL_DIR, data.get('preModelName') + '.pt')


    ## 2、根据task_type来获取数据路径
    if task_type == 'Detect':
        images_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Detection', 'images')
        labels_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Detection', 'labels')
        coco_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Detection', 'coco8.yaml')
    elif task_type == 'Segment':
        images_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Segment', 'images')
        labels_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Segment', 'labels')
        coco_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Segment', 'coco8.yaml')
    elif task_type == 'Classification':
        images_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Classification', 'images')
        labels_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Classification', 'lables')
        coco_path = os.path.join(LOCAL_DATASET_DIR, data.get('user'), data.get('datasetName'), 'Classification', 'coco8.yaml')
    else:
        return jsonify({'status': 'error', 'message': '任务类型错误'})
    

    ## 3、交给ray去进行训练
    future = training.remote(training_id, minio_folder_name, coco_path, pre_model_path, images_path, labels_path, epochs, img_size, batch_size)
    ray_manager.submit_training_task(training_id, future)
    ray_manager.tasks[training_id] = future
    return jsonify({'status': 'training started'})

@app.route('/delete_temp_dir', methods=['POST', 'OPTIONS'])
def delete_dataset_dir():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.get_json()
    user = data.get('params').get('user')
    dataset_name = data.get('params').get('datasetName')
    is_deleted = delete_temp_dir(os.path.join(LOCAL_DATASET_DIR, user, dataset_name))
    if is_deleted:
        return jsonify({'success': is_deleted, 'message': '删除成功'}), 200
    else:
        return jsonify({'success': is_deleted, 'message': '删除失败'}), 400

@app.route('/stop_training_ray_task', methods=['GET'])
def stop_training_ray_task():
    data = request.get_json()  # 获取传来的数据
    training_id = int(data.get('training_id'))
    ray_manager.cancel_task(training_id)
    return jsonify({'success': True, 'message': f"任务{training_id}已停止训练"})
    
@app.route('/temp_test', methods=['GET'])
def temp_test():
    result = main_test()
    return jsonify({'success': True, 'message': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8096)  # 监听所有可用的IP地址

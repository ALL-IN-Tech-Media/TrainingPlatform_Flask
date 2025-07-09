from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import ray
import os
import zipfile
import shutil
from config import BUCKET_NAME, LOCAL_DATASET_DIR, LOCAL_PRE_MODEL_DIR
from minio_tools import delete_temp_dir, download_minio_folder
from concurrent.futures import ThreadPoolExecutor
from NLP.text_generation_train_api import text_generation_fine_tuning_lora, text_generation_fine_tuning_all, estimate_gpu_memory_text_generation, text_generation_fine_tuning_dora
from NLP.embedding_train_api import embedding_fine_tuning_all, estimate_gpu_memory_embedding
from tools import validate_and_update_yaml_fields
from ray_manage import RayTaskManage
import torch
ray.init(ignore_reinit_error=True, include_dashboard=True, dashboard_host="0.0.0.0")  # 初始化Ray
from ray.util.state import summarize_tasks, list_workers, get_worker, get_log, list_tasks, get_task
import docker
import socket

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})  # 允许所有来源的请求并支持凭证

executor = ThreadPoolExecutor(max_workers=10)
ray_manager = RayTaskManage()

client = docker.from_env()

# 设置图片文件夹的路径
IMAGE_FOLDER = '/Users/liuzizhen/Projects/AIData/'  # 替换为您的图片文件夹路径
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

training_container_map = {}

def find_free_port(start=10000, end=20000):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found")

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
    
    ## 判断数据集是否存在
    dataset_path = os.path.join(os.path.join(os.path.join(LOCAL_DATASET_DIR, str(user_id)), str(dataset_id)), 'datasets', "data.json")
    if not os.path.exists(dataset_path):
        return jsonify({'code': 500, 'message': '数据集不存在', 'data':{}})
    
    ## 判断预训练模型是否存在
    model_path = os.path.join(LOCAL_PRE_MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'code': 500, 'message': '模型不存在', 'data':{}})
    
    ## 计算任务大概所需要的显存
    model_memory = estimate_gpu_memory_text_generation(model_name, training_type, model_path, max_length, batch_size)
    print(f"模型大概所需要的显存：{model_memory}MB")    
    free_mem, total_mem = torch.cuda.mem_get_info(gpu[0])
    print(f"剩余显存: {free_mem / 1024**2:.2f} MB")
    if (free_mem / 1024**2) < model_memory:
        return jsonify({'code': 500, 'message': f'显存不足，当前剩余显存：{free_mem / 1024**2:.2f} MB，所需显存：{model_memory:.2f} MB', 'data':{}})

    # 创建ray任务去训练
    if training_type == 'fine_tuning_all':
        future = text_generation_fine_tuning_all.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    elif training_type == 'fine_tuning_lora':
        future = text_generation_fine_tuning_lora.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    elif training_type == 'fine_tuning_dora':
        future = text_generation_fine_tuning_dora.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    elif training_type == 'fine_tuning_qlora':
        return jsonify({'code': 500, 'message': 'QLoRA训练暂未支持', 'data':{}})
        # future = fine_tuning_qlora.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    else:
        return jsonify({'code': 500, 'message': '任务类型错误', 'data':{}})
    ray_manager.submit_training_task(training_id, future)
    ray_manager.tasks[training_id] = future

    pid, message = ray_manager.get_task_pid(training_id)
    print(f"pid: {pid}, message: {message}")
    return jsonify({'code': 200, 'message': "创建训练任务成功，" + message, 'data':{'pid': pid}})

@app.route('/text_generation_fine_tuning', methods=['POST'])
def text_generation_fine_tuning():
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
    
    ## 判断数据集是否存在
    dataset_path = os.path.join(os.path.join(os.path.join(LOCAL_DATASET_DIR, str(user_id)), str(dataset_id)), 'datasets', "data.json")
    if not os.path.exists(dataset_path):
        return jsonify({'code': 500, 'message': '数据集不存在', 'data':{}})
    
    ## 判断预训练模型是否存在
    model_path = os.path.join(LOCAL_PRE_MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'code': 500, 'message': '模型不存在', 'data':{}})
    
    ## 计算任务大概所需要的显存
    model_memory = estimate_gpu_memory_text_generation(model_name, training_type, model_path, max_length, batch_size)
    print(f"模型大概所需要的显存：{model_memory}MB")    
    free_mem, total_mem = torch.cuda.mem_get_info(gpu[0])
    print(f"剩余显存: {free_mem / 1024**2:.2f} MB")
    if (free_mem / 1024**2) < model_memory:
        return jsonify({'code': 500, 'message': f'显存不足，当前剩余显存：{free_mem / 1024**2:.2f} MB，所需显存：{model_memory:.2f} MB', 'data':{}})

    # 创建ray任务去训练
    if training_type == 'fine_tuning_all':
        future = text_generation_fine_tuning_all.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    elif training_type == 'fine_tuning_lora':
        future = text_generation_fine_tuning_lora.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    elif training_type == 'fine_tuning_dora':
        future = text_generation_fine_tuning_dora.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    elif training_type == 'fine_tuning_qlora':
        return jsonify({'code': 500, 'message': 'QLoRA训练暂未支持', 'data':{}})
        # future = fine_tuning_qlora.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    else:
        return jsonify({'code': 500, 'message': '任务类型错误', 'data':{}})
    ray_manager.submit_training_task(training_id, future)
    ray_manager.tasks[training_id] = future

    pid, message = ray_manager.get_task_pid(training_id)
    return jsonify({'code': 200, 'message': "创建训练任务成功，" + message, 'data':{'pid': pid}})

@app.route('/embedding_fine_tuning', methods=['POST'])
def embedding_fine_tuning():
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
    
    ## 判断数据集是否存在
    dataset_path = os.path.join(os.path.join(os.path.join(LOCAL_DATASET_DIR, str(user_id)), str(dataset_id)), 'datasets', "data.json")
    if not os.path.exists(dataset_path):
        return jsonify({'code': 500, 'message': '数据集不存在', 'data':{}})
    
    ## 判断预训练模型是否存在
    model_path = os.path.join(LOCAL_PRE_MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'code': 500, 'message': '模型不存在', 'data':{}})
    
    ## 计算任务大概所需要的显存
    model_memory = estimate_gpu_memory_embedding(model_name, model_path, max_length, batch_size)
    print(f"模型大概所需要的显存：{model_memory}MB")    
    free_mem, total_mem = torch.cuda.mem_get_info(gpu[0])
    print(f"剩余显存: {free_mem / 1024**2:.2f} MB")
    if (free_mem / 1024**2) < model_memory:
        return jsonify({'code': 500, 'message': f'显存不足，当前剩余显存：{free_mem / 1024**2:.2f} MB，所需显存：{model_memory:.2f} MB', 'data':{}})

    # 创建ray任务去训练
    if training_type == 'fine_tuning_all':
        future = embedding_fine_tuning_all.remote(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu)
    else:
        return jsonify({'code': 500, 'message': '任务类型错误', 'data':{}})
    ray_manager.submit_training_task(training_id, future)
    ray_manager.tasks[training_id] = future

    return jsonify({'code': 200, 'message': "创建训练任务成功"})

@app.route('/get_training_pid', methods=['GET'])
def get_training_pid():
    try:
        data = request.get_json()
        training_id = int(data.get('training_id'))
        pid, message = ray_manager.get_task_pid(training_id)
        return jsonify({'code': 200, 'message': message, 'data':{'pid': pid}})
    except Exception as e:
        return jsonify({'code': 500, 'message': '获取失败：' + str(e), 'data':{}})

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
    
@app.route('/deploy_model', methods=['POST'])
def deploy_model():
    data = request.get_json()
    user_id = data.get('user_id')
    dataset_id = data.get('dataset_id')
    training_id = data.get('training_id')
    gpu = data.get('gpu')
    max_num_seqs = data.get('max_num_seqs')
    max_model_len = data.get('max_model_len')
    max_num_batched_tokens = data.get('max_num_batched_tokens')
    container_name = data.get('container_name', f"vllm_{training_id}_best")

    if not all([training_id, gpu, max_num_seqs, max_model_len, max_num_batched_tokens]):
        return jsonify({'code': 400, 'message': '缺少参数', 'data': {}}), 400

    # 查找本地未被占用端口
    try:
        host_port = find_free_port(10000, 20000)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'找不到可用端口: {e}', 'data': {}}), 500

    # 检查并删除同名容器
    try:
        old_container = client.containers.get(container_name)
        print(f"发现同名容器 {container_name}，正在停止并删除...")
        old_container.stop(timeout=10)
        old_container.remove()
        print(f"已删除旧容器 {container_name}")
    except Exception as e:
        return jsonify({'code': 500, 'message': f'删除旧容器时出错: {e}', 'data': {}}), 500

    try:
        command = [
            "--model", f"/home/ooin/ooin_training/NLPData/{user_id}/{dataset_id}/save_models/{training_id}/best",
            "--tensor-parallel-size", str(len(gpu)),
            "--served-model-name", f"{training_id}_best",
            "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(max_num_seqs),
            "--max-num-batched-tokens", str(max_num_batched_tokens)
        ]
        device_requests = [docker.types.DeviceRequest(device_ids=[str(x) for x in gpu], capabilities=[["gpu"]])]
        container = client.containers.run(
            image="vllm/vllm-openai:v0.7.3",
            command=command,
            detach=True,
            device_requests=device_requests,
            ports={'8000/tcp': host_port},  # 动态分配端口
            volumes={
                '/home/ooin/ooin_training/model_factory': {
                    'bind': '/home/ooin/ooin_training/model_factory',
                    'mode': 'rw'
                },
                '/home/ooin/ooin_training/NLPData': {
                    'bind': '/home/ooin/ooin_training/NLPData',
                    'mode': 'rw'
                }
            },
            name=container_name
        )
        # 记录映射
        training_container_map[training_id] = {
            'container_id': container.id,
            'container_name': container_name,
            'host_port': host_port
        }
        return jsonify({'code': 200, 'message': '容器已启动', 'data': {'container_id': container.id, 'container_name': container_name, 'host_port': host_port}})
    except Exception as e:
        return jsonify({'code': 500, 'message': f'启动容器时出错: {e}', 'data': {}}), 500

@app.route('/embedding_deploy_model', methods=['POST'])
def embedding_deploy_model():
    data = request.get_json()
    user_id = data.get('user_id')
    dataset_id = data.get('dataset_id')
    training_id = data.get('training_id')
    gpu = data.get('gpu')
    container_name = data.get('container_name', f"embedding_{training_id}_api")

    if not all([training_id, model_path, gpu]):
        return jsonify({'code': 400, 'message': '缺少参数', 'data': {}}), 400

    # 查找本地未被占用端口
    try:
        host_port = find_free_port(10000, 20000)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'找不到可用端口: {e}', 'data': {}}), 500

    # 检查并删除同名容器
    try:
        old_container = client.containers.get(container_name)
        print(f"发现同名容器 {container_name}，正在停止并删除...")
        old_container.stop(timeout=10)
        old_container.remove()
        print(f"已删除旧容器 {container_name}")
    except docker.errors.NotFound:
        pass
    except Exception as e:
        return jsonify({'code': 500, 'message': f'删除旧容器时出错: {e}', 'data': {}}), 500

    try:
        device_requests = [docker.types.DeviceRequest(device_ids=[str(x) for x in gpu], capabilities=[["gpu"]])]
        container = client.containers.run(
            image="embedding-api:latest",
            detach=True,
            device_requests=device_requests,
            ports={'8000/tcp': host_port},
            volumes={
                '/home/ooin/ooin_training/model_factory': {
                    'bind': '/home/ooin/ooin_training/model_factory',
                    'mode': 'rw'
                },
                '/home/ooin/ooin_training/NLPData': {
                    'bind': '/home/ooin/ooin_training/NLPData',
                    'mode': 'rw'
                }
            },
            environment={
                'EMBEDDING_MODEL_PATH': model_path
            },
            name=container_name
        )
        # 记录映射
        training_container_map[training_id] = {
            'container_id': container.id,
            'container_name': container_name,
            'host_port': host_port
        }
        return jsonify({'code': 200, 'message': '容器已启动', 'data': {'container_id': container.id, 'container_name': container_name, 'host_port': host_port}})
    except Exception as e:
        return jsonify({'code': 500, 'message': f'启动容器时出错: {e}', 'data': {}}), 500

@app.route('/stop_deploy_model', methods=['POST'])
def stop_deploy_model():
    data = request.get_json()
    training_id = data.get('training_id')
    info = training_container_map.get(training_id)
    if not info:
        return jsonify({'code': 404, 'message': '未找到对应容器', 'data': {}}), 404
    try:
        container = client.containers.get(info['container_name'])
        container.stop(timeout=10)
        container.remove()
        del training_container_map[training_id]
        return jsonify({'code': 200, 'message': '停止容器成功'})
    except Exception as e:
        return jsonify({'code': 500, 'message': f'停止容器失败: {e}', 'data': {}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8096)  # 监听所有可用的IP地址

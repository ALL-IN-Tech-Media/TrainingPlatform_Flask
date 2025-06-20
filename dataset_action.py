import requests
from config import AIDJANGO_URL
def update_database_status(training_id, status):
    """向 Django 后端发送请求以更新数据库状态"""
    url = AIDJANGO_URL + "/training/update_training_status/"  # 替换为实际的 Django API URL
    data = {
        "id": training_id, 
        "status": status
    }
    try:
        response = requests.get(url, params=data)
        response.raise_for_status()  # 检查请求是否成功
        print("数据库状态更新成功:", response.json())
    except requests.exceptions.RequestException as e:
        print("更新数据库状态时出错:", e)


def insert_training_epoch_status(training_id, epoch_number, map_50, map_95, precision, recall):
    """向 Django 后端发送请求以更新数据库状态"""
    url = AIDJANGO_URL + "/training/insert_training_epoch_status/"  # 替换为实际的 Django API URL
    data = {
        "training_id": training_id,
        "epoch_number": epoch_number,
        "map_50": map_50,
        "map_95": map_95,
        "precision": precision,
        "recall": recall
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 检查请求是否成功
        print(f"插入记录成功:", response.json())
    except requests.exceptions.RequestException as e:
        print("插入记录时出错:", e)


if __name__ == "__main__":
    # update_database_status(54, "开始训练")
    insert_training_epoch_status(73, 1, 50.0, 50.0, 50.0, 50.0)
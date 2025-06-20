import yaml
import os
import xml.etree.ElementTree as ET

def validate_and_update_yaml_fields(yaml_file_path, train_path, val_path):
    categories = {}
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # 更新或创建 'train' 字段
    data['train'] = train_path.replace("\\", "\\\\")

    # 更新或创建 'val' 字段
    data['val'] = val_path.replace("\\", "\\\\")

    # 检查 'names' 字段是否存在
    if 'names' not in data:
        return False, categories


    # 检查 'names' 字段下是否至少有一个类别，
    for key, value in data['names'].items():
        if not value:  # 如果类别为空
            return False, categories
        else:
            categories = data["names"]
    
    # 检查 'names' 字段是否符合规范
    keys = sorted(categories.keys())  # 对键进行排序
    for expected_key in range(len(keys)):
        if keys[expected_key] != expected_key:
            return False, categories  # 如果不符合要求，返回错误

    # 将更新后的数据写回 YAML 文件
    with open(yaml_file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True)  # 允许写入 Unicode 字符

    return True, categories

def convert_xml_to_txt(categories, labels_path):
    """将 XML 文件转换为 YOLOv5 格式的 TXT 文件"""
    try:
        for folder in ['train', 'val']:
            folder_path = os.path.join(labels_path, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith('.xml'):
                    xml_file = os.path.join(folder_path, filename)
                    txt_file = os.path.join(folder_path, filename.replace('.xml', '.txt'))
                    
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    with open(txt_file, 'w') as f:
                        for obj in root.findall('object'):
                            class_name = obj.find('name').text
                            class_id = next((id for id, name in categories.items() if name == class_name), -1)  # 获取类别 ID
                            
                            if class_id == -1:
                                continue  # 如果类别不存在，则跳过

                            # 获取边界框坐标
                            bndbox = obj.find('bndbox')
                            xmin = int(bndbox.find('xmin').text)
                            ymin = int(bndbox.find('ymin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymax = int(bndbox.find('ymax').text)

                            # 计算 YOLO 格式的中心坐标和宽高
                            x_center = (xmin + xmax) / 2
                            y_center = (ymin + ymax) / 2
                            width = xmax - xmin
                            height = ymax - ymin

                            # 归一化到 [0, 1] 范围
                            img_width = 640  # 替换为实际图像宽度
                            img_height = 640  # 替换为实际图像高度
                            x_center /= img_width
                            y_center /= img_height
                            width /= img_width
                            height /= img_height

                            # 写入 TXT 文件
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                    # 替换原 XML 文件
                    os.remove(xml_file)  # 删除原 XML 文件
                    os.rename(txt_file, xml_file.replace('.xml', '.txt'))  # 将 TXT 文件重命名为原 XML 文件名
    except Exception as e:
        return False
    finally:
        return True
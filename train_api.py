import os
from config import BUCKET_NAME, LOCAL_DATASET_DIR, LOCAL_PRE_MODEL_DIR
from ultralytics import YOLO
from tools import validate_and_update_yaml_fields, convert_xml_to_txt
from minio_tools import download_minio_folder
from dataset_action import update_database_status, insert_training_epoch_loss
import ray
from yolov5.train import really_training

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import traceback
import time


model_total_params = {
    "Qwen2.5-0.5B-Instruct": 494032768,
    "Qwen2.5-1.5B-Instruct": 1543714304,
    "Qwen2.5-7B-Instruct": 617005056,
    "Qwen2.5-14B-Instruct": 1234010112,
    "Qwen2.5-32B-Instruct": 2468020224,
}

# 4. 数据集预处理
class InstructionDataset(torch.utils.data.Dataset):
    """
    Qwen2.5/PEFT微调专用数据集，支持messages格式，自动分词和label处理。
    """
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

# 5. 验证函数
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count if count > 0 else 0

def estimate_gpu_memory_realistic(
    model_name: str,
    train_type: str,
    model_path: str,
    max_length: int,
    batch_size: int,
    lora_rank: int = 8,
    lora_target_modules: list = ["q_proj", "v_proj"],
    precision: str = "auto",
) -> float:
    if train_type == "fine_tuning_lora":
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
            hidden_size = config["hidden_size"]
            num_hidden_layers = config["num_hidden_layers"]
            num_attention_heads = config["num_attention_heads"]
            vocab_size = config["vocab_size"]

        # 1. 模型参数显存（MB，和全量微调一样）
        m1 = model_total_params[model_name] * 2 / (1024**2) + 800

        # 2. LoRA参数显存（MB，极小）
        lora_params_per_module = hidden_size * lora_rank * 2
        lora_total_params = len(lora_target_modules) * lora_params_per_module
        lora_memory_MB = lora_total_params * 2 / (1024**2)  # float16

        # 3. LoRA优化器状态显存（MB，极小，Adam假设4字节/参数）
        lora_optimizer_MB = lora_total_params * 4 / (1024**2)

        # 4. 激活显存（MB，和全量微调一样）
        m3 = (batch_size * max_length * hidden_size * (34 + 5 * num_attention_heads * max_length / hidden_size) * num_hidden_layers / (1024**2))
        attention_memory_MB = batch_size * num_hidden_layers * num_attention_heads * max_length * max_length * 2 / (1024**2)

        # 5. 总显存（MB）
        total_memory = m1 + lora_memory_MB + lora_optimizer_MB + m3 + attention_memory_MB
        total_memory = total_memory + (batch_size - 1) * 350

        return round(total_memory * 1.1, 2)  # 加10%冗余
    elif train_type == "fine_tuning_all":
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
            hidden_size = config["hidden_size"]
            num_hidden_layers = config["num_hidden_layers"]
            num_attention_heads = config["num_attention_heads"]
            vocab_size = config["vocab_size"]
        m1 = model_total_params[model_name] * 2 / (1024**2)
        m2 = (m1) * 2 + (m1) * 2 
        m3 = (batch_size * max_length * hidden_size * (34 + 5 * num_attention_heads * max_length / hidden_size) * num_hidden_layers / (1024**2))
        attention_memory_MB = batch_size * num_hidden_layers * num_attention_heads * max_length * max_length * 2 / (1024**2)
        total_memory = m2 + m3 + attention_memory_MB + m1
        total_memory = total_memory + (batch_size - 1) * 350
        return round(total_memory + total_memory * 0.1, 2)
    elif train_type == "fine_tuning_qlora":
        return 0.0
    elif train_type == "fine_tuning_dora":
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
            hidden_size = config["hidden_size"]
            num_hidden_layers = config["num_hidden_layers"]
            num_attention_heads = config["num_attention_heads"]
            vocab_size = config["vocab_size"]

        # 1. 模型参数显存（MB，和全量微调一样）
        m1 = model_total_params[model_name] * 2 / (1024**2) + 800

        # 2. LoRA参数显存（MB，极小）
        lora_params_per_module = hidden_size * lora_rank * 2
        lora_total_params = len(lora_target_modules) * lora_params_per_module
        lora_memory_MB = lora_total_params * 2 / (1024**2)  # float16

        # 3. LoRA优化器状态显存（MB，极小，Adam假设4字节/参数）
        lora_optimizer_MB = lora_total_params * 4 / (1024**2)

        # 4. 激活显存（MB，和全量微调一样）
        m3 = (batch_size * max_length * hidden_size * (34 + 5 * num_attention_heads * max_length / hidden_size) * num_hidden_layers / (1024**2))
        attention_memory_MB = batch_size * num_hidden_layers * num_attention_heads * max_length * max_length * 2 / (1024**2)

        # 5. 总显存（MB）
        total_memory = m1 + lora_memory_MB + lora_optimizer_MB + m3 + attention_memory_MB
        total_memory = total_memory + (batch_size - 1) * 350

        return round(total_memory * 1.1, 2)  # 加10%冗余
    else:
        return 0.0

# @ray.remote(max_calls=1, num_gpus=0.2)
@ray.remote(max_calls=1)
def fine_tuning_lora(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu):
    time.sleep(1)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in gpu)  # 动态设置本worker可见的GPU
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 加载预训练模型和分词器
        model_name = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

        # 2. LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]  # Qwen/llama常用
        )
        model = get_peft_model(model, lora_config)
        print("LoRA trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # 3. 加载指令数据集（JSONL格式）
        def load_instruction_data(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        
        train_data_full = load_instruction_data(dataset_path)
        split_idx = int(len(train_data_full) * 0.7)
        train_data = train_data_full[:split_idx]
        val_data = train_data_full[split_idx:]

        train_dataset = InstructionDataset(train_data, tokenizer, max_length=max_length)
        val_dataset = InstructionDataset(val_data, tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 6. 训练循环
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        update_database_status(training_id, "训练中")
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                    
            avg_loss = total_loss/len(train_loader)
            val_loss = evaluate(model, val_loader, device)
            insert_training_epoch_loss(training_id, epoch+1, round(avg_loss, 4), round(val_loss, 4))

            # 每save_epoch轮保存一次模型
            if (epoch + 1) % save_epoch == 0:
                save_path = os.path.join(save_dir, str(training_id), f"epoch{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                model.module.save_pretrained(save_path) if hasattr(model, "module") else model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[保存] LoRA权重和分词器已保存到: {save_path}")
        
        update_database_status(training_id, "训练完成")
        return "training_success"
    except Exception as e:
        update_database_status(training_id, "训练异常")
        traceback.print_exc()
        return f"training_failed: {str(e)}"

@ray.remote(max_calls=1)
def fine_tuning_all(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu):
    time.sleep(1)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in gpu)  # 动态设置本worker可见的GPU
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 加载预训练模型和分词器
        model_name = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

        # 3. 加载指令数据集（JSONL格式）
        def load_instruction_data(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        
        train_data_full = load_instruction_data(dataset_path)
        split_idx = int(len(train_data_full) * 0.7)
        train_data = train_data_full[:split_idx]
        val_data = train_data_full[split_idx:]

        train_dataset = InstructionDataset(train_data, tokenizer, max_length=max_length)
        val_dataset = InstructionDataset(val_data, tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 6. 训练循环
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        update_database_status(training_id, "训练中")
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                    
            avg_loss = total_loss/len(train_loader)
            val_loss = evaluate(model, val_loader, device)
            insert_training_epoch_loss(training_id, epoch+1, round(avg_loss, 4), round(val_loss, 4))

            # 每save_epoch轮保存一次模型
            if (epoch + 1) % save_epoch == 0:
                save_path = os.path.join(save_dir, str(training_id), f"epoch{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                model.module.save_pretrained(save_path) if hasattr(model, "module") else model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[保存] LoRA权重和分词器已保存到: {save_path}")
        
        update_database_status(training_id, "训练完成")
        return "training_success"
    except Exception as e:
        update_database_status(training_id, "训练异常")
        traceback.print_exc()
        return f"training_failed: {str(e)}"

@ray.remote(max_calls=1)
def fine_tuning_dora(training_id, user_id, dataset_path, save_dir, model_path, epochs, batch_size, max_length, save_epoch, gpu):
    time.sleep(1)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in gpu)  # 动态设置本worker可见的GPU
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 加载预训练模型和分词器
        model_name = model_path
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

        # 2. LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],  # Qwen/llama常用
            use_dora=True,
        )
        model = get_peft_model(model, lora_config)
        print("LoRA trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # 3. 加载指令数据集（JSONL格式）
        def load_instruction_data(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        
        train_data_full = load_instruction_data(dataset_path)
        split_idx = int(len(train_data_full) * 0.7)
        train_data = train_data_full[:split_idx]
        val_data = train_data_full[split_idx:]

        train_dataset = InstructionDataset(train_data, tokenizer, max_length=max_length)
        val_dataset = InstructionDataset(val_data, tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 6. 训练循环
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        update_database_status(training_id, "训练中")
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                    
            avg_loss = total_loss/len(train_loader)
            val_loss = evaluate(model, val_loader, device)
            insert_training_epoch_loss(training_id, epoch+1, round(avg_loss, 4), round(val_loss, 4))

            # 每save_epoch轮保存一次模型
            if (epoch + 1) % save_epoch == 0:
                save_path = os.path.join(save_dir, str(training_id), f"epoch{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                model.module.save_pretrained(save_path) if hasattr(model, "module") else model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"[保存] LoRA权重和分词器已保存到: {save_path}")
        
        update_database_status(training_id, "训练完成")
        return "training_success"
    except Exception as e:
        print("Exception occurred during fine-tuning:")
        traceback.print_exc()
        return f"training_failed: {str(e)}"

if __name__ == "__main__":
    pass

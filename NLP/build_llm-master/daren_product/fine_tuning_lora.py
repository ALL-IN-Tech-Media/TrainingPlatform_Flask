import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import os

# 明确指定使用单卡或多卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 修改为你可用的GPU编号

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载预训练模型和分词器
model_name = "/home/ooin/lzz/study/build_llm/huggingface/Qwen2.5-0.5B-Instruct"  # 修改为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

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

# 多卡支持
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
    model = torch.nn.DataParallel(model)
model.to(device)

# 3. 加载指令数据集（JSONL格式）
def load_instruction_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data_full = load_instruction_data("/home/ooin/lzz/study/build_llm/datasets/cnews/train.jsonl")
split_idx = int(len(train_data_full) * 0.85)
train_data = train_data_full[:split_idx]
val_data = train_data_full[split_idx:]

# 4. 数据集预处理
class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = tokenizer.apply_chat_template(
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
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

train_dataset = InstructionDataset(train_data, tokenizer, max_length=1024)
val_dataset = InstructionDataset(val_data, tokenizer, max_length=1024)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 5. 验证函数
def evaluate(model, loader):
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

# 6. 训练循环
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(3):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
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
    print(f"Epoch {epoch+1} average loss: {avg_loss}")
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

# 7. 保存LoRA权重
save_dir = "/home/ooin/lzz/study/build_llm/fine_tuning/qwen2.5-instruct-lora"
os.makedirs(save_dir, exist_ok=True)
model.module.save_pretrained(save_dir) if hasattr(model, "module") else model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"LoRA权重和分词器已保存到: {save_dir}") 
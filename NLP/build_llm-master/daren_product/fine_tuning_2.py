import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# 明确指定使用单卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 只使用第一张卡
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载预训练模型和分词器
model_name = "/home/ooin/lzz/study/build_llm/huggingface/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
    model = torch.nn.DataParallel(model)
# model.to(device)

# 2. 加载指令数据集（JSONL格式）
def load_instruction_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 示例数据路径
train_data_full = load_instruction_data("/home/ooin/lzz/study/build_llm/datasets/cnews/train.jsonl")
split_idx = int(len(train_data_full) * 0.85)
train_data = train_data_full[:split_idx]
val_data = train_data_full[split_idx:]

# 3. 数据集预处理（适配messages格式）
class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 使用apply_chat_template处理messages格式
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        # print(text)  # 可选：调试时打印
        
        # 分词处理
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 创建labels（全部非pad token都参与loss）
        labels = encoding['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

dataset = InstructionDataset(train_data, tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=8)

# 4. 验证函数
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

# 5. 自定义训练循环
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(3):  # 3个epoch
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        # 将数据移到GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss/len(loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss}")
    
    # 验证
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

# 6. 保存模型
model.save_pretrained("/home/ooin/lzz/study/build_llm/fine_tuning/qwen2.5-instruct-finetuned")
tokenizer.save_pretrained("/home/ooin/lzz/study/build_llm/fine_tuning/qwen2.5-instruct-finetuned")
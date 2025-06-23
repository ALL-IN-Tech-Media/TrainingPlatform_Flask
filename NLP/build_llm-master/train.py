import tiktoken
from study_simple.split_words import create_dataloader_v1
from main import generate_text_simple, text_to_token_ids, token_ids_to_text, GPTModel
import torch
from torch.optim.lr_scheduler import StepLR
from datasets import load_dataset
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def belle_dataset_to_text(ds, max_samples=None):
    samples = ds["train"]
    if max_samples:
        samples = samples.select(range(max_samples))
    
    conversations = []
    for sample in samples:
        instr = sample.get("instruction", "").strip()
        output = sample.get("output", "").strip()
        if instr and output:
            conversations.append(f"用户：{instr}\n助手：{output}")
    
    return "\n\n".join(conversations)

GPT_CONFIG_124M = {
    "vocab_size": 50257, # 词汇表大小
    "context_length": 1024, # 上下文长度
    "emb_dim": 768, # 嵌入维度
    "n_heads": 12, # 注意力头数
    "n_layers": 12, # 层数
    "drop_rate": 0.1, # Dropout率
    "qkv_bias": False # Query-Key-Value偏置
}

# 计算损失
def calc_loss_batch(input_batch, target_batch, model, device):
    # 将输入和目标批次数据移动到指定的设备（如GPU或CPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) #A
    # 前向传播：将输入批次送入模型，得到预测的logits（未归一化的概率分布）
    logits = model(input_batch)
    # 计算交叉熵损失：
    # - torch.nn.functional.cross_entropy：计算预测与目标之间的交叉熵损失
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), # - logits.flatten(0, 1)：将logits展平成二维（batch*seq_len, vocab_size），适应损失函数输入格式
        target_batch.flatten() # - target_batch.flatten()：将目标展平成一维（batch*seq_len），与logits对应
    )
    # 返回当前批次的损失值
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    # 初始化总损失为0
    total_loss = 0.
    # 如果数据加载器为空，返回NaN
    if len(data_loader) == 0:
        return float("nan")
    # 如果未指定num_batches，则默认遍历整个数据集
    elif num_batches is None:
        num_batches = len(data_loader) #A
    else:
        # 如果指定了num_batches，则最多只遍历num_batches个批次
        num_batches = min(num_batches, len(data_loader)) #B
    # 遍历数据加载器，按批次计算损失
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 累加损失值
            total_loss += loss.item() #C
        else:
            # 达到指定批次数后停止
            break
    # 返回平均损失
    return total_loss / num_batches

# 评估模型
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter) # 用训练集前eval_iter个批次的损失来表示这一小轮的整体损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter) # 用验证集前eval_iter个批次的损失来表示这一小轮的整体损失
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# 训练模型
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    # 用于记录训练损失、验证损失和已见token数的列表
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1  # tokens_seen累计已训练token数，global_step为全局步数
    with open("epoch_losses.txt", "w") as loss_file:
        for epoch in range(num_epochs):
            model.train()  # 设置模型为训练模式
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # 梯度清零
                loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次损失
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新模型参数
                tokens_seen += input_batch.numel()  # 累加本批次token数
                global_step += 1  # 步数加一
                if global_step % eval_freq == 0:  # 每隔eval_freq步进行一次评估
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)  # 评估当前模型
                    train_losses.append(train_loss)  # 记录训练损失
                    val_losses.append(val_loss)  # 记录验证损失
                    track_tokens_seen.append(tokens_seen)  # 记录已见token数
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )  # 打印当前损失
                    # 写入loss到文件
                    loss_file.write(f"Step {global_step}: Train loss {train_loss:.6f}, Val loss {val_loss:.6f}\n")
                    loss_file.flush()
            generate_and_print_sample(model, tokenizer, device, start_context)  # 每个epoch结束后生成一段文本样例
            # 每轮结束后写入loss到文件
            if len(train_losses) > 0 and len(val_losses) > 0:
                loss_file.write(f"Epoch {epoch+1}: Train loss {train_losses[-1]:.6f}, Val loss {val_losses[-1]:.6f}\n")
                loss_file.flush()
            # 每10轮保存一次模型
            if (epoch + 1) % 2 == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "tokens_seen": track_tokens_seen
                }, f"model_and_optimizer_epoch{epoch+1}.pth")
    # 训练结束后再保存一次最终模型
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "model_and_optimizer.pth")
    return train_losses, val_losses, track_tokens_seen  # 返回损失和token统计

# 加载预训练权重进行训练
def train1():
    import urllib.request
    from study_simple.gpt_download import download_and_load_gpt2

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    print("settings:", settings)
    print("params:", params)

# 使用the-verdict.txt进行训练
def train2():
    file_path = "text_data.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    tokenizer = tiktoken.get_encoding("gpt2") # 初始化一个编码器
    total_characters = len(text_data) # 计算文本总字符数
    total_tokens = len(tokenizer.encode(text_data)) # 计算文本总token数 (一个词算作一个token，当然也可能存在重复的token，所以只有5145个token)


    train_ratio = 0.8 # 训练集比例
    split_idx = int(train_ratio * len(text_data)) # 计算分割索引
    train_data = text_data[:split_idx] # 分割文本，前70%作为训练集
    val_data = text_data[split_idx:] # 分割文本，后30%作为验证集

    # 创建数据加载器
    # torch.Size([2, 256]) torch.Size([2, 256]) （256是上下文长度，2是batch_size）（一个训练样本对应一个target）
    # 例如一段文本："Every effort moves you"，如果一个训练样本就是"Every effort"，其对应的target就是"effort moves" 这里"Every effort" token为2，我们实际训练时的token为256
    train_loader = create_dataloader_v1(train_data, batch_size=2, 
                                        max_length=GPT_CONFIG_124M["context_length"], 
                                        stride=GPT_CONFIG_124M["context_length"], 
                                        shuffle=True, 
                                        drop_last=True, 
                                        num_workers=0)
    val_loader = create_dataloader_v1(val_data, batch_size=2, 
                                        max_length=GPT_CONFIG_124M["context_length"], 
                                        stride=GPT_CONFIG_124M["context_length"], 
                                        shuffle=False, 
                                        drop_last=False, 
                                        num_workers=0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    num_epochs = 100
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=20,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

# 使用cnews数据集进行语义预训练
def train3():
    train_path = "datasets/cnews_lang/train.txt"
    val_path = "datasets/cnews_lang/val.txt"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = "\n".join([line.strip() for line in f if line.strip()])
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = "\n".join([line.strip() for line in f if line.strip()])

    tokenizer = tiktoken.get_encoding("cl100k_base")  # 仍可用，后期可换中文BPE tokenizer

    train_loader = create_dataloader_v1(train_data, batch_size=10, 
                                        max_length=GPT_CONFIG_124M["context_length"], 
                                        stride=GPT_CONFIG_124M["context_length"], 
                                        shuffle=True, 
                                        drop_last=True, 
                                        num_workers=0)
    val_loader = create_dataloader_v1(val_data, batch_size=4, 
                                        max_length=GPT_CONFIG_124M["context_length"], 
                                        stride=GPT_CONFIG_124M["context_length"], 
                                        shuffle=False, 
                                        drop_last=False, 
                                        num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 100
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=200, eval_iter=20,
        start_context="新闻：", tokenizer=tokenizer
    )

if __name__ == "__main__":
    train3()

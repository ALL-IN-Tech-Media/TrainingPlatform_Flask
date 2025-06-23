import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257, # 词汇表大小
    "context_length": 1024, # 上下文长度
    "emb_dim": 768, # 嵌入维度
    "n_heads": 12, # 注意力头数
    "n_layers": 12, # 层数
    "drop_rate": 0.1, # Dropout率
    "qkv_bias": False # Query-Key-Value偏置
}

# GPTModel 关系图
# ├── tok_emb (Embedding)
# ├── pos_emb (Embedding)
# ├── drop_emb (Dropout)
# ├── trf_blocks (Sequential)
# │   └── TransformerBlock (×n_layers)
# │       ├── att (MultiHeadAttention)
# │       ├── ff (FeedForward)
# │       ├── norm1 (LayerNorm)
# │       ├── norm2 (LayerNorm)
# │       └── drop_shortcut (Dropout)
# ├── final_norm (LayerNorm)
# └── out_head (Linear)

# 数据流动示例
# 1、输入token IDs (batch, seq_len)
# 2、经过词嵌入和位置嵌入 → (batch, seq_len, emb_dim)
# 3、通过多个TransformerBlock处理
# 4、最终归一化后输出logits (batch, seq_len, vocab_size)

# 归一化层
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数，提升数值稳定性
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数 γ，初始为全1，shape为(emb_dim,)
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 可学习的平移参数 β，初始为全0，shape为(emb_dim,)

    def forward(self, x):
        # x: 输入张量，形状通常为 (batch, seq_len, emb_dim) 或 (batch, emb_dim)
        mean = x.mean(dim=-1, keepdim=True)  # 对最后一维（特征维，也就是768个维度）求均值，保持维度不变，shape和x一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 对最后一维求方差，unbiased=False更适合深度学习
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 标准化：减均值除以标准差
        return self.scale * norm_x + self.shift  # 仿射变换，恢复模型表达能力

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 检查输出维度能否被头数整除
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out  # 总输出维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_out // num_heads  # 每个头的维度
        # 定义Q、K、V的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out) # 多头输出拼接后再通过一个线性层
        self.dropout = nn.Dropout(dropout) # dropout层
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)) # 注册上三角掩码（防止看到未来信息）

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取batch大小、序列长度、输入维度
        # 计算Q、K、V
        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        values = self.W_value(x)  # (b, num_tokens, d_out)

        # 拆分多头 (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 交换维度，方便多头并行 (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) # 计算注意力分数 (b, num_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # 取出与当前序列长度匹配的掩码
        attn_scores.masked_fill_(mask_bool, -torch.inf) # 掩码未来信息，设为-inf
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # 缩放后softmax归一化，得到注意力权重
        attn_weights = self.dropout(attn_weights) # dropout
        context_vec = (attn_weights @ values).transpose(1, 2) # 用注意力权重加权value，得到上下文向量 (b, num_heads, num_tokens, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # 拼接所有头 (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec) # 输出线性层融合各头信息
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 多头自注意力层，输入输出维度都是emb_dim
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = LayerNorm(cfg["emb_dim"]) # 第一层归一化（用于注意力前）
        self.ff = FeedForward(cfg) # 前馈全连接网络（简单说就是激活层，引入非线性变换）
        self.norm2 = LayerNorm(cfg["emb_dim"]) # 第二层归一化（用于前馈前）
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"]) # 残差连接后的dropout

    def forward(self, x):
        shortcut = x  # 保存输入，后面做残差连接
        x = self.norm1(x)  # 先做归一化 输入（2，4，768）对每个token的768个维度做归一化，使其均值为0，方差为1. 输出仍为（2，4，768）
                           # 归一化让每个token的特征分布更加稳定，避免梯度消失或爆炸。实际上归一化后经过很多次线性变换后，数值也可能变的极端，但是我们每一轮都会归一化........
        x = self.att(x)    # 多头自注意力
        x = self.drop_shortcut(x)  # dropout防止过拟合
        x = x + shortcut   # 残差连接：加回原始输入
        shortcut = x       # 再次保存当前x，后面做第二次残差
        x = self.norm2(x)  # 第二次归一化
        x = self.ff(x)     # 前馈全连接网络（就是引入非线性变换，让模型可以学习到更复杂的特征）
        x = self.drop_shortcut(x)  # dropout
        x = x + shortcut   # 残差连接：加回前面的shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token id是分词表中的索引，从0开始。例如gpt-2的词表有50257个词，所以vocab_size=50257
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # 词嵌入层，将token id映射为向量 (vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # 位置嵌入层，编码每个token在序列中的位置信息 (context_length, emb_dim)
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # 对嵌入加dropout，防止过拟合
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] ) # 堆叠多个TransformerBlock，形成深层结构
        self.final_norm = LayerNorm(cfg["emb_dim"]) # 最后一层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # 输出头，将emb_dim映射回词表大小，用于生成下一个token的概率分布
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 输入形状(batch, seq_len) (2, 4)
        tok_embeds = self.tok_emb(in_idx) # 词嵌入 (batch, seq_len, emb_dim) (2, 4, 768)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) # 位置嵌入 (seq_len, emb_dim)，自动广播到(batch, seq_len, emb_dim)
        x = tok_embeds + pos_embeds # 词嵌入和位置嵌入相加，获得每个token的最终输入表示 (2, 4, 768)代表两段文本，每段4个token，每个token的维度是768
        x = self.drop_emb(x)  # dropout防止过拟合 （2, 4, 768）
        x = self.trf_blocks(x) # 经过多层TransformerBlock （2, 4, 768）
        x = self.final_norm(x) # 最后一层归一化
        logits = self.out_head(x) # 输出头，得到每个位置上每个词的logits (batch, seq_len, vocab_size) (2, 4, 50257)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # 取出当前序列中最后context_size个token，作为模型输入（防止序列过长超出模型最大长度）例[1,2,3,4,5,6,7,8,9,10] 取最后4个token作为模型输入[6,7,8,9]
        with torch.no_grad():
            logits = model(idx_cond)  # (batch, seq_len, vocab_size)
        logits = logits[:, -1, :]  # 取出每个序列最后一个位置的logits（即下一个token的预测分布）
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # 将新的token id加到后面作为下一轮的输入 (batch, seq_len+1)
    return idx

# 将文本进行编码，返回token id的tensor
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

# 将token id的tensor转换为文本
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)

    
    model = GPTModel(GPT_CONFIG_124M)
    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)
    
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    # print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print("encoded_tensor.shape:", encoded_tensor.shape)
    model.eval() # 禁用dropout，因为我们不在训练模型
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # print("out:", out)
    # print("Output:", out)
    # print("Output length:", len(out[0]))
    # decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    # print(decoded_text)
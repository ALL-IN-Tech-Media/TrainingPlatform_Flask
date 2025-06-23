import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257, # 词汇表大小
    "context_length": 1024, # 上下文长度
    "emb_dim": 768, # 嵌入维度
    "n_heads": 12, # 注意力头数
    "n_layers": 12, # 层数
    "drop_rate": 0.1, # Dropout率
    "qkv_bias": False # Query-Key-Value偏置
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
        *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]) #A
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) #B
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module): #C
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x): #D
        return x

class DummyLayerNorm(nn.Module): #E
    def __init__(self, normalized_shape, eps=1e-5): #F
        super().__init__()
    def forward(self, x):
        return x

# 归一化层
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数，提升数值稳定性
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数 γ，初始为全1，shape为(emb_dim,)
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 可学习的平移参数 β，初始为全0，shape为(emb_dim,)

    def forward(self, x):
        # x: 输入张量，形状通常为 (batch, seq_len, emb_dim) 或 (batch, emb_dim)
        mean = x.mean(dim=-1, keepdim=True)  # 对最后一维（特征维）求均值，保持维度不变，shape和x一致
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
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

# 归一化示例
def test1():
    torch.manual_seed(123)
    batch_example = torch.rand(2, 5)
    layer = nn.Sequential(
        nn.Linear(5, 6),
        nn.ReLU()
    )
    out = layer(batch_example)
    print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("均值:\n", mean)
    print("方差:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("归一化的层输出:\n", out_norm)
    print("均值:\n", mean)
    print("方差:\n", var)

if __name__ == "__main__":
    # test1()

    # ffn = FeedForward(GPT_CONFIG_124M)
    # x = torch.rand(2, 3, 768) #A
    # out = ffn(x)
    # print(out.shape)

    # torch.manual_seed(123)
    # batch_example = torch.rand(2, 5)
    # ln = LayerNorm(emb_dim=5)
    # out_ln = ln(batch_example)
    # mean = out_ln.mean(dim=-1, keepdim=True)
    # var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)
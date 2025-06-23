import torch
import torch.nn as nn
from dummymodel import LayerNorm, GPT_CONFIG_124M
from study_simple.transforms import TransformerBlock
import tiktoken

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token id是分词表中的索引，从0开始。例如gpt-2的词表有50257个词，所以vocab_size=50257
        # 词嵌入层，将token id映射为向量 (vocab_size, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入层，编码每个token在序列中的位置信息 (context_length, emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 对嵌入加dropout，防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 堆叠多个TransformerBlock，形成深层结构
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] )
        # 最后一层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出头，将emb_dim映射回词表大小，用于生成下一个token的概率分布
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 输入形状(batch, seq_len)
        # 词嵌入 (batch, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        # 位置嵌入 (seq_len, emb_dim)，自动广播到(batch, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 词嵌入和位置嵌入相加，获得每个token的最终输入表示
        x = tok_embeds + pos_embeds
        # dropout防止过拟合
        x = self.drop_emb(x)
        # 经过多层TransformerBlock
        x = self.trf_blocks(x)
        # 最后一层归一化
        x = self.final_norm(x)
        # 输出头，得到每个位置上每个词的logits (batch, seq_len, vocab_size)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size): #A
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] 
        probas = torch.softmax(logits, dim=-1) #D
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) #E
        idx = torch.cat((idx, idx_next), dim=1) #F
    return idx


if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    # print(batch)

    
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)
    
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
    print("encoded_tensor.shape:", encoded_tensor.shape)
    model.eval() # 禁用dropout，因为我们不在训练模型
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("out:", out)
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
import torch
import torch.nn as nn
from causal_attention import CausalAttention, CausalAttention_v1, CausalAttention_v2

class MultiHeadAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

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
        # 多头输出拼接后再通过一个线性层
        self.out_proj = nn.Linear(d_out, d_out)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册上三角掩码（防止看到未来信息）
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
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
        # 计算注意力分数 (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        # 取出与当前序列长度匹配的掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 掩码未来信息，设为-inf
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 缩放后softmax归一化，得到注意力权重
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        # dropout
        attn_weights = self.dropout(attn_weights)
        # 用注意力权重加权value，得到上下文向量 (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 拼接所有头 (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 输出线性层融合各头信息
        context_vec = self.out_proj(context_vec)
        return context_vec

if __name__ == "__main__":
    inputs = torch.tensor( [[0.43, 0.15, 0.89], # Your (x^1)
                            [0.55, 0.87, 0.66], # journey (x^2)
                            [0.57, 0.85, 0.64], # starts (x^3)
                            [0.22, 0.58, 0.33], # with (x^4)
                            [0.77, 0.25, 0.10], # one (x^5)
                            [0.05, 0.80, 0.55]] # step (x^6)
                        )
    mult = MultiHeadAttention_v1(d_in=3, d_out=2, context_length=inputs.shape[0], dropout=0.5, num_heads=2)
    context_vecs = mult.forward(inputs)
    print("context_vecs:", context_vecs)
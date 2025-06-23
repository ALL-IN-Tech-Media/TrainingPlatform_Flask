import torch
import torch.nn as nn

# 添加因果注意力掩码
class CausalAttention_v1(nn.Module):
    def __init__(self, d_in, d_out,
    qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # x: 输入张量，形状为 (序列长度, 输入特征维度)
        # 通过线性层计算key、query、value
        keys = self.W_key(x)      # (seq_len, d_out)
        queries = self.W_query(x) # (seq_len, d_out)
        values = self.W_value(x)  # (seq_len, d_out)
        # 计算注意力分数（点积注意力），结果为 (seq_len, seq_len)
        attn_scores = queries @ keys.T
        # 对分数进行缩放并softmax归一化，得到注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # 设置因果掩码
        context_length = attn_scores.shape[0]
        masked_simple = torch.tril(torch.ones(context_length, context_length))
        masked_simple = masked_simple * attn_weights # '**'代表对应位置相乘，'@'代表矩阵乘法运算
        
        # 重新归一化
        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        masked_simple_norm = masked_simple / row_sums
        attn_weights = masked_simple_norm

        # 用注意力权重加权所有value，得到每个token的上下文向量
        context_vec = attn_weights @ values
        return context_vec

# 添加因果注意力掩码和dropout
class CausalAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(0.5)
        self.mask = torch.tril(torch.ones(context_length, context_length))

    def forward(self, x):
        # x: 输入张量，形状为 (序列长度, 输入特征维度)
        # 通过线性层计算key、query、value
        keys = self.W_key(x)      # (seq_len, d_out)
        queries = self.W_query(x) # (seq_len, d_out)
        values = self.W_value(x)  # (seq_len, d_out)
        # 计算注意力分数（点积注意力），结果为 (seq_len, seq_len)
        attn_scores = queries @ keys.T
        # 对分数进行缩放
        attn_scores = attn_scores / keys.shape[-1]**0.5
        # 使用因果掩码（上三角为1，mask掉未来信息）
        seq_len = x.shape[0]
        mask = self.mask[:seq_len, :seq_len].bool() # 0/1掩码变为bool掩码

        attn_scores = attn_scores.masked_fill(~mask, float('-inf')) # 将掩码位置全部设为无穷小“-inf”

        # softmax归一化
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # dropout
        attn_weights = self.dropout(attn_weights)
        # 用注意力权重加权所有value，得到每个token的上下文向量
        context_vec = attn_weights @ values
        return context_vec

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.5, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length), diagonal=1)) 
    def forward(self, x):
        num_tokens, d_in = x.shape 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T #C
        attn_scores.masked_fill_( #D
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    inputs = torch.tensor( [[0.43, 0.15, 0.89], # Your (x^1)
                            [0.55, 0.87, 0.66], # journey (x^2)
                            [0.57, 0.85, 0.64], # starts (x^3)
                            [0.22, 0.58, 0.33], # with (x^4)
                            [0.77, 0.25, 0.10], # one (x^5)
                            [0.05, 0.80, 0.55]] # step (x^6)
                        )
    # print(inputs.shape) # (6, 3)
    torch.manual_seed(123)
    ca_v2 = CausalAttention_v2(d_in=3, d_out=2, context_length=inputs.shape[0])
    context_vecs_v2 = ca_v2.forward(inputs)
    torch.manual_seed(123)
    ca = CausalAttention(d_in=3, d_out=2, context_length=inputs.shape[0])
    context_vecs = ca.forward(inputs)
    print("context_v2:", context_vecs_v2)
    print("context:", context_vecs)

    
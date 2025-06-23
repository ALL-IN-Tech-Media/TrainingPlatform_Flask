import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # 初始化三个参数矩阵，分别用于生成query、key、value
        # d_in: 输入特征维度，d_out: 输出特征维度
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        # x: 输入张量，形状为 (序列长度, 输入特征维度)
        # 计算所有token的key、query、value
        keys = x @ self.W_key      # (seq_len, d_out)
        queries = x @ self.W_query # (seq_len, d_out)
        values = x @ self.W_value  # (seq_len, d_out)
        # 计算注意力分数（点积注意力），结果为 (seq_len, seq_len)
        attn_scores = queries @ keys.T
        # 对分数进行缩放并softmax归一化，得到注意力权重
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        # 用注意力权重加权所有value，得到每个token的上下文向量
        context_vecs = attn_weights @ values
        return context_vecs

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # 使用nn.Linear实现query、key、value的线性变换，可以选择是否加偏置
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
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        # 用注意力权重加权所有value，得到每个token的上下文向量
        context_vec = attn_weights @ values
        return context_vec
        

    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
    qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
        "mask", torch.triu(torch.ones(context_length,
        context_length),
        diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x) #C
        values = self.W_value(x) #C
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)#D
        keys = keys.transpose(1, 2) #E
        queries = queries.transpose(1, 2) #E
        values = values.transpose(1, 2) #E
        attn_scores = queries @ keys.transpose(2, 3) #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #G
        attn_scores.masked_fill_(mask_bool, -torch.inf) #H
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) #I
        #J
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) #K
        return context_vec

def test1():
    text = "Your journey starts with one step"
    inputs = torch.tensor( [[0.43, 0.15, 0.89], # Your (x^1)
                            [0.55, 0.87, 0.66], # journey (x^2)
                            [0.57, 0.85, 0.64], # starts (x^3)
                            [0.22, 0.58, 0.33], # with (x^4)
                            [0.77, 0.25, 0.10], # one (x^5)
                            [0.05, 0.80, 0.55]] # step (x^6)
                        )

    x_2 = inputs[1] # x_2: tensor([0.5500, 0.8700, 0.6600])
    d_in = inputs.shape[1] # d_in: 3 代表维度
    d_out = 2

    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    print("------------初始化的三个权重参数矩阵---------------")
    print(W_query)
    print(W_key)
    print(W_value)

    # query_2 = x_2 @ W_query
    # key_2 = x_2 @ W_key
    # value_2 = x_2 @ W_value
    # print("------------计算x_2的query,key,value---------------")
    # print(query_2)
    # print(key_2)
    # print(value_2)

    queries = inputs @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value
    print("------------计算所有向量的keys,values---------------")
    print("queries:", queries)
    print("keys:", keys)
    print("values:", values)

    # attn_scores_2 = query_2 @ keys.T
    # print("------------计算x_2的query和所有key的注意力得分---------------")
    # print(attn_scores_2)

    attn_scores = queries @ keys.T
    print("------------计算所有向量query和key的注意力得分---------------")
    print("attn_scores:", attn_scores)

    # d_k = keys.shape[-1]
    # attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
    # print("------------计算x_2的query和所有key的注意力权重---------------")
    # print(attn_weights_2)

    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print("------------计算所有向量query和key的注意力权重---------------")
    print("attn_weights:", attn_weights)

    # context_vec_2 = attn_weights_2 @ values
    # print("------------计算x_2的query和所有key的上下文向量---------------")
    # print(context_vec_2)

    context_vec = attn_weights @ values
    print("------------计算所有向量query和key的上下文向量---------------")
    print("context_vec:", context_vec)

if __name__ == "__main__":
    # test1()
    inputs = torch.tensor( [[0.43, 0.15, 0.89], # Your (x^1)
                            [0.55, 0.87, 0.66], # journey (x^2)
                            [0.57, 0.85, 0.64], # starts (x^3)
                            [0.22, 0.58, 0.33], # with (x^4)
                            [0.77, 0.25, 0.10], # one (x^5)
                            [0.05, 0.80, 0.55]] # step (x^6)
                        )
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v2(d_in=3, d_out=2)
    # context_vecs = sa_v1.forward(inputs)
    # print("context:", context_vecs)
    # sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
    # queries = sa_v1.W_query(inputs)
    # print(queries)

    # keys = sa_v2.W_key(inputs)
    # print(keys)

    # attn_scores = queries @ keys.T
    # attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    # print(attn_weights)

    # context_length = attn_scores.shape[0]
    # mask_simple = torch.tril(torch.ones(context_length, context_length))
    # masked_simple = attn_weights*mask_simple
    # print(masked_simple)

    # ## 使用掩码矩阵后归一化
    # row_sums = masked_simple.sum(dim=-1, keepdim=True)
    # attn_weights = masked_simple / row_sums
    # print(attn_weights)


    


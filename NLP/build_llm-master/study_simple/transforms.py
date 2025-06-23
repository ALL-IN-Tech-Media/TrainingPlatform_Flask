from study_simple.multihead_attention import MultiHeadAttention
from dummymodel import FeedForward, LayerNorm
import torch
import torch.nn as nn
from dummymodel import GPT_CONFIG_124M

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
            qkv_bias=cfg["qkv_bias"])
        # 前馈全连接网络（简单说就是激活层，引入非线性变换）
        self.ff = FeedForward(cfg)
        # 第一层归一化（用于注意力前）
        self.norm1 = LayerNorm(cfg["emb_dim"])
        # 第二层归一化（用于前馈前）
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 残差连接后的dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x  # 保存输入，后面做残差连接
        x = self.norm1(x)  # 先做归一化
        x = self.att(x)    # 多头自注意力
        x = self.drop_shortcut(x)  # dropout防止过拟合
        x = x + shortcut   # 残差连接：加回原始输入
        shortcut = x       # 再次保存当前x，后面做第二次残差
        x = self.norm2(x)  # 第二次归一化
        x = self.ff(x)     # 前馈全连接网络
        x = self.drop_shortcut(x)  # dropout
        x = x + shortcut   # 残差连接：加回前面的shortcut
        return x
    

if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print(output.shape)
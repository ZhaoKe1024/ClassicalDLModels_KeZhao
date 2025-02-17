#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/17 16:35
# @Author: ZhaoKe
# @File : mha.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        # 转置最内层两个维度，其他维度广播
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
                 np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """

    def __init__(self, embed_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = 64
        self.n_heads = n_heads
        self.W_Q = nn.Linear(embed_dim, self.d_k * n_heads,
                             bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(embed_dim, self.d_k * n_heads, bias=False)
        self.W_V = nn.Linear(embed_dim, self.d_v * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * self.d_v, embed_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.n_heads, self.d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        print("MyMHA attn mask:", attn_mask.shape)
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_v)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(output.device)(output + residual), attn

# 定义多头注意力模块
d_model = 512
num_heads = 8
# 假设输入张量 X 的形状为 [batch_size, seq_len, d_model]
batch_size = 32
seq_len = 128

print("My MHA:")
X = torch.randn(batch_size, seq_len, d_model)  # 随机初始化输入张量 [2, 5, 8]
# 打印输入张量的形状
print("输入张量 X 的形状:", X.shape)
attn_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.float32)
# attn_mask[0:8, 0, 3:, :] = -float('inf')  # 对于第一个序列，掩盖索引3及之后的部分
# # attn_mask[1, 0, :, :] 保持为0，因为第二个序列没有需要掩盖的部分
multihead_attn = MultiHeadAttention(embed_dim=d_model, n_heads=num_heads)
# 执行多头注意力机制
attn_output, attn_output_weights = multihead_attn(X, X, X, attn_mask=attn_mask)  # 自注意力机制，Q, K, V 相同
# 打印输出张量的形状
print("输出张量 attn_output 的形状:", attn_output.shape)  # [2, 5, 8]
print("注意力权重 attn_output_weights 的形状:", attn_output_weights.shape)  # [5, 2, 2]

print("Torch's MHA:")
X = torch.randn(seq_len, batch_size, d_model)  # 随机初始化输入张量 [2, 5, 8]
multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
# # 但由于 PyTorch 的 nn.MultiheadAttention 期望的 attn_mask 形状是 [batch_size, num_heads, seq_len, seq_len]
# # 我们需要调整 attn_mask 的形状
# attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
# 执行多头注意力机制
# 参数和输出形状：https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
attn_output, attn_output_weights = multihead_attn(X, X, X, average_attn_weights=False)  # 自注意力机制，Q, K, V 相同
# 打印输出张量的形状
print("输出张量 attn_output 的形状:", attn_output.shape)  # [2, 5, 8]
print("注意力权重 attn_output_weights 的形状:", attn_output_weights.shape)  # [5, 2, 2]

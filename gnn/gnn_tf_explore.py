#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/17 16:20
# @Author: ZhaoKe
# @File : gnn_tf_explore.py
# @Software: PyCharm
import torch
import torch.nn as nn


class GNNBasedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # GNN参数化消息函数
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 生成Q,K,V [batch_size, seq_len, d_model]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 拆分为多头 [batch_size, num_heads, seq_len, depth]
        Q = Q.view(*Q.shape[:2], self.num_heads, self.depth).permute(0, 2, 1, 3)
        K = K.view(*K.shape[:2], self.num_heads, self.depth).permute(0, 2, 1, 3)
        V = V.view(*V.shape[:2], self.num_heads, self.depth).permute(0, 2, 1, 3)

        # 消息传递（动态边权计算）
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.depth))
        weights = torch.softmax(scores, dim=-1)

        # 邻域聚合（加权求和）
        context = torch.matmul(weights, V)

        # 节点更新（拼接+线性变换）
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.shape[:2], self.d_model)
        return context


class AttentionBasedGNNLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, x, adj_matrix):
        # x: [seq_len, batch_size, d_model]
        # adj_matrix: [batch_size, seq_len, seq_len] (二值掩码或权重矩阵)

        # 注意力计算（应用图结构约束）
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=adj_matrix  # 关键：用邻接矩阵限制注意力范围
        )
        return attn_output


if __name__ == '__main__':
    # 使用示例：
    gnn_layer = AttentionBasedGNNLayer(d_model=64, num_heads=4)
    node_features = torch.randn(10, 32, 64)  # [seq_len, batch_size, d_model]
    adj = (torch.rand(32, 10, 10) > 0.5).float()  # 生成随机邻接矩阵

    output = gnn_layer(node_features, adj)


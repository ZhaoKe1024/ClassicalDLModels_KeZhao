#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/17 16:20
# @Author: ZhaoKe
# @File : gnn_tf_explore.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv  # GNN layer with Transformer-style attention


# From DeepSeek-R1
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


# From DeepSeek-R1
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


# From ChatGPT DeepSearch
def MHSA_by_PyG():
    # Define a small fully-connected graph (e.g., a sentence of N words represented as nodes)
    N = 4  # number of nodes (words)
    d_in = 8   # input feature dimension (embedding size per word)
    d_out = 8  # output feature dimension per head
    num_heads = 2

    # Random node features to represent word embeddings for this example:
    x = torch.rand((N, d_in))  # shape [N, d_in]

    # Define fully-connected edges (every pair of distinct nodes is connected).
    # We'll create edge indices for an undirected fully-connected graph (excluding self-loops for now, as TransformerConv can add them by default if needed).
    edge_indices = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_indices.append([i, j])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    # edge_index is a [2, E] tensor listing all directed edges in the graph.

    # Create a PyG Data object for the graph
    graph = Data(x=x, edge_index=edge_index)

    # Initialize a TransformerConv layer.
    # in_channels = d_in, out_channels = d_out, heads = num_heads.
    # By default, TransformerConv will concatenate heads (so output dim = d_out * heads).
    conv = TransformerConv(in_channels=d_in, out_channels=d_out, heads=num_heads, concat=True)

    # Apply the TransformerConv layer to get new node features
    new_x = conv(graph.x, graph.edge_index)
    print("Output features shape:", new_x.shape)
    # The output new_x has shape [N, d_out * num_heads] because heads are concatenated.
    # In this example, that would be [4, 8*2] = [4, 16] if concat=True.


# From ChatGPT DeepSearch
def GNN_by_MHSA():
    # Example graph: 4 nodes in a chain (0 -- 1 -- 2 -- 3).
    N = 4
    d_model = 8  # feature dimension (must be divisible by number of heads for convenience)
    num_heads = 2
    d_head = d_model // num_heads  # dimension per head

    # Node feature matrix (random initialization for demonstration)
    X = torch.rand((N, d_model))  # shape [N, d_model]

    # Adjacency matrix for a chain graph (no self-connections for this example):
    adj = torch.tensor([
        [0, 1, 0, 0],  # Node 0 is connected to 1
        [1, 0, 1, 0],  # Node 1 connected to 0 and 2
        [0, 1, 0, 1],  # Node 2 connected to 1 and 3
        [0, 0, 1, 0]  # Node 3 connected to 2
    ], dtype=torch.float32)
    # In this adjacency, adj[i,j] = 1 if there's an edge j->i (assuming undirected chain, it's symmetric).

    # Define learnable weight matrices for multi-head attention (random init).
    # These correspond to W^Q, W^K, W^V, and W^O in Transformer formulas.
    W_Q = torch.nn.Linear(d_model, d_model, bias=False)
    W_K = torch.nn.Linear(d_model, d_model, bias=False)
    W_V = torch.nn.Linear(d_model, d_model, bias=False)
    W_O = torch.nn.Linear(d_model, d_model, bias=False)

    # Compute Queries, Keys, Values for each node
    Q = W_Q(X)  # shape [N, d_model]
    K = W_K(X)  # shape [N, d_model]
    V = W_V(X)  # shape [N, d_model]

    # Reshape Q, K, V to separate heads (shape -> [num_heads, N, d_head])
    Q = Q.view(N, num_heads, d_head).transpose(0, 1)  # [num_heads, N, d_head]
    K = K.view(N, num_heads, d_head).transpose(0, 1)  # [num_heads, N, d_head]
    V = V.view(N, num_heads, d_head).transpose(0, 1)  # [num_heads, N, d_head]

    # Compute attention scores for each head:
    # scores[h, i, j] = dot_product of Q_i_h and K_j_h (for head h)
    scores = torch.matmul(Q, K.transpose(-1, -2))  # shape [num_heads, N, N]
    scores = scores / (d_head ** 0.5)  # scale by sqrt(d_head)

    # Apply mask so that a node only attends to its neighbors (based on adjacency matrix).
    # We will use the adjacency as a mask: if adj[i,j] = 0 (no edge from j to i), set score to -inf.
    # First, expand adjacency matrix to [num_heads, N, N] (same mask for all heads):
    mask = adj.unsqueeze(0).expand(num_heads, -1, -1)  # shape [num_heads, N, N]
    # Use masked_fill to put a large negative value where mask == 0:
    scores = scores.masked_fill(mask == 0, float('-inf'))

    # Compute attention weights with softmax (along the last dimension j for each i):
    attn_weights = F.softmax(scores, dim=-1)  # shape [num_heads, N, N]

    # Use the attention weights to get weighted values for each head:
    # For each head h and node i: out_h[i] = sum_j (attn_weights[h,i,j] * V[h,j])
    head_outputs = torch.bmm(attn_weights, V)  # shape [num_heads, N, d_head]

    # Concatenate heads back together for each node:
    head_outputs = head_outputs.transpose(0, 1).contiguous()  # [N, num_heads, d_head]
    head_outputs = head_outputs.view(N, num_heads * d_head)  # [N, d_model]  (since num_heads*d_head = d_model)

    # Final linear projection to mix head outputs (as in standard multi-head attention)
    output = W_O(head_outputs)  # shape [N, d_model]

    print("Output feature vectors for each node:\n", output)


if __name__ == '__main__':
    MHSA_by_PyG()
    print("============================")
    GNN_by_MHSA()

    # # 使用示例：
    # gnn_layer = AttentionBasedGNNLayer(d_model=64, num_heads=4)
    # node_features = torch.randn(10, 32, 64)  # [seq_len, batch_size, d_model]
    # adj = (torch.rand(32, 10, 10) > 0.5).float()  # 生成随机邻接矩阵
    #
    # output = gnn_layer(node_features, adj)


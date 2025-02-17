#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/17 15:23
# @Author: ZhaoKe
# @File : gcn.py
# @Software: PyCharm
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_udg_data():
    # 示例特征矩阵 (4个节点, 每个节点有3个特征)
    features = torch.tensor([[1, 2, 0.3], [9, 8, 7.5], [7, 6, 5.5], [5.12, 5.87, 5]], dtype=torch.float)
    print(features.size(0), features.size(1))
    # 示例边索引 (无向图边: 0-1, 1-2, 2-3, 3-0)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                               [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    # 确保边索引是无向的
    edge_index = to_undirected(edge_index, num_nodes=features.size(0))

    # 创建图数据对象
    graph = Data(x=features, edge_index=edge_index)
    return graph


def build_trainer():
    dataset = Planetoid(root='C:/data/CiteSeer', name='CiteSeer', transform=NormalizeFeatures())
    print(len(dataset), '\n', dataset[0], '\n', dataset[0].is_directed())
    print(dataset.y)
    print(dataset.num_features, dataset.num_classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # data = Planetoid(root='/data/CiteSeer', name='CiteSeer')
    model = GCN(dataset.num_node_features, 32, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    print(model)

    # model = GCN(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()

    # graph = get_udg_data()
    # labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)  # 示例标签

    # 训练模型（这里仅为了演示，通常你需要更多的训练步骤和更复杂的数据预处理）
    pbar = tqdm(total=200, desc="Epoch ")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(x=data.x, edge_index=data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        tr_loss = loss.item()

        model.eval()
        logits = model(x=data.x, edge_index=data.edge_index)
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        print(f'Epoch: {epoch:03d}, Loss: {tr_loss:.4f}, Train Acc: {accs[0]:.4f}, '
              f'Val Acc: {accs[1]:.4f}, Test Acc: {accs[2]:.4f}')
        pbar.update(1)
    print(f'Final Test Accuracy: {accs[2]:.4f}')
    # # 提取节点嵌入
    # model.eval()
    # with torch.no_grad():
    #     embeddings = model(graph)
    #
    # # 使用KMeans进行聚类
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings.numpy())
    # cluster_labels = kmeans.labels_
    #
    # print("Cluster labels:", cluster_labels)


if __name__ == '__main__':
    build_trainer()

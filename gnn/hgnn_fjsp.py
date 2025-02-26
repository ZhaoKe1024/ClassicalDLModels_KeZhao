#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/2/26 14:48
# @Author: ZhaoKe
# @File : hgnn_fjsp.py
# @Software: PyCharm

# reference: WenSong, Flexible_Job-Shop_Scheduling_via_Graph_Neural_Network_and_Deep_Reinforcement_Learning
# reference: https://github.com/songwenas12/fjsp-drl
import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.distributions import Categorical


class GATedge(nn.Module):
    '''
    Machine node embedding
    '''

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        (in_size_ope=6, in_size_ma=3), out_size_ma=8, num_heads=[1, 1], dropout=0.1
        in_feats=(6, 3),               out_feats=8,
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]  # 工序数 6
        self._in_dst_feats = in_feats[1]  # 机器数 3
        self._out_feats = out_feats  # 机器嵌入 8

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                1, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation  # ＥＬＵ

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, ope_ma_adj_batch, batch_idxes, feat):
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to
        #   W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            if not hasattr(self, 'fc_src'):
                self.fc_src, self.fc_dst = self.fc, self.fc
            feat_src = self.fc_src(h_src)
            feat_dst = self.fc_dst(h_dst)
        else:
            # Deprecated in final experiment
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        # 边的特征，长度为1
        feat_edge = self.fc_edge(feat[2].unsqueeze(-1))

        # Calculate attention coefficients
        # 分别计算工序点、机器点、边的注意力
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        ee = (feat_edge * self.attn_e).sum(dim=-1).unsqueeze(-1)
        # 邻接矩阵乘以注意力系数
        el_add_ee = ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(a)
        ekk = self.leaky_relu(er + er)

        # Normalize attention coefficients
        mask = torch.cat((ope_ma_adj_batch[batch_idxes].unsqueeze(-1) == 1,
                          torch.full(size=(ope_ma_adj_batch[batch_idxes].size(0), 1,
                                           ope_ma_adj_batch[batch_idxes].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        # 通过mask之后进行softmax可以对不需要的数据得到0值
        e[~mask] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)  # e经过softmax就是alpha
        alpha_ijk = alpha[..., :-1, :]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # Calculate an return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)  # 拼接再乘矩阵，和分别乘矩阵再相加，一样
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)  # 注意力系数e_ijk
        b = torch.sum(a, dim=-3)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)  # 注意力系数
        nu_k_prime = torch.sigmoid(b + c)
        return nu_k_prime


class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''

    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        # print("shape adj feat:", adj.shape, feat.shape)
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        # (batch, ope, ope/mas, )
        # print("shape a: ", a.shape)
        b = torch.sum(a, dim=-2)
        c = self.project(b)
        return c


class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''

    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        # shape:(1, 194, 6); opes shape:(1, 10, 3)
        # machine, opes, opes, opes
        h = (feats[1], feats[0], feats[0], feats[0])

        # Identity matrix for self-loop of nodes
        # print("eye shape:", torch.eye(feats[0].size(-2), dtype=torch.int64).unsqueeze(0).shape)
        # print("ope_pre_adj shape:", ope_pre_adj_batch[batch_idxes])

        # 自身邻接矩阵，batch个堆叠单位矩阵
        # (batch, opes_num, opes_num)
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])

        # 四个邻接矩阵：和机器、和前序、和后续、和自身。矩阵元素都是1
        # Calculate an return operation embedding
        # print("adj shapes: ", ope_ma_adj_batch[batch_idxes].shape, ope_pre_adj_batch[batch_idxes].shape,
        #        ope_sub_adj_batch[batch_idxes].shape, self_adj.shape)
        adj = (ope_ma_adj_batch[batch_idxes], ope_pre_adj_batch[batch_idxes],
               ope_sub_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        # input shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma * 2 + out_size_ope * 2]
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class HGNNScheduler(nn.Module):
    def __init__(self):
        super(HGNNScheduler, self).__init__()
        model_paras = {"device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       "in_size_ma": 3,
                       "out_size_ma": 8,
                       "in_size_ope": 6,
                       "out_size_ope": 8,
                       "hidden_size_ope": 128,
                       "n_latent_actor": 64,
                       "n_latent_critic": 64,
                       "n_hidden_actor": 3,
                       "n_hidden_critic": 3,
                       "action_dim": 1,
                       "num_heads": [1, 1],
                       "dropout": 0.0}
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs

        model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
        model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic

        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        # Machine node embedding
        self.get_machines = nn.ModuleList()
        # (in_size_ope=6, in_size_ma=3), out_size_ma=8, num_heads=[1, 1], dropout=0.1
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                         self.dropout, self.dropout, activation=F.elu))
        for i in range(1, len(self.num_heads)):
            # (out_size_ope=8, out_size_ma=8), out_size_ma=8, num_heads[1]=1, dropout=0.1, dropout=0.1
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                             self.dropout, self.dropout, activation=F.elu))

        # Operation node embedding
        """ return torch(size=(batch, out_size_ope=8))"""
        self.get_operations = nn.ModuleList()
        # [out_size_ma=8, in_size_ope=6, in_size_ope=6, in_size_ope=6], hidden_size_ope=128, out_size_ope=8
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads) - 1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        """
        model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
        out_size_ma=8, out_size_ope=8, -->, actor_in_dim = 32
        n_hidden_actor=3, actor_dim=32, n_latent_actor=64, action_dim=1
        MLP_Actor: nn.ModuleList([nn.Linear(32, 64),
                                nn.Linear(64, 64),
                                nn.Linear(64, 1)])
        h = torch.tanh(self.linears[layer](h))
        self.linears[layer](h)
        """
        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)

        """
        model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
        critic_in_dim = 16
        n_hidden_critic=3, critic_dim=16, n_latent_critic=644, output_dim=1
        MLP_Critic: nn.ModuleList([nn.Linear(16, 64),
                                nn.Linear(64, 64),
                                nn.Linear(64, 1)])
        h = torch.tanh(self.linears[layer](h))
        return self.linears[layer](h)
        """
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def get_embeddings(self, state, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        # 对operation feature，machines feature，proc time执行标准化
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        # norm_opes = (copy.deepcopy(features[0]))
        # norm_mas = (copy.deepcopy(features[1]))
        # norm_proc = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        # self.num_heads = [1, 1]
        # print(f"shape ope, machine, time: {features[0].shape}, {features[1].shape}, {features[2].shape}")
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            """
                # Machine node embedding
                self.get_machines = nn.ModuleList()
                # (in_size_ope=6, in_size_ma=3), out_size_ma=8, num_heads=[1, 1], dropout=0.1
                self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                                 self.dropout, self.dropout, activation=F.elu))
                for i in range(1, len(self.num_heads)):
                    # (out_size_ope=8, out_size_ma=8), out_size_ma=8, num_heads[1]=1, dropout=0.1, dropout=0.1
                    self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                                     self.dropout, self.dropout, activation=F.elu))
                h_mas = attention embedding of machine
            """
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, features)

            # print(f">-------------head {i}--------------->")
            """features: (ope, mas, proc_time)"""
            features = (features[0], h_mas, features[2])
            # print(f"shape ope, machine, time: {features[0].shape}, {features[1].shape}, {features[2].shape}")
            # Second Stage, operation node embedding
            # shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, features)
            "embedding of operations"
            features = (h_opes, features[1], features[2])
            # print(f"shape ope, machine, time: {features[0].shape}, {features[1].shape}, {features[2].shape}")
        return features

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''

    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances
        # print("batch_size=batch_idxes: ", batch_size)
        # print("batch_idxes:", batch_idxes)
        # print("size:", batch_idxes.size())
        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                # 第[1]维求平均，因为工序数不同而有所填充，因此要得到每个instance的工序数索引
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))  # 对batch工序求平均
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])  # 得到坐标列表
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)  # 对整个batch的所有工序在每个机器上的执行时间，求标准化
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        # 对operation feature，machines feature，proc time执行标准化
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        # print("features shape:", features[0].shape, features[1].shape, features[2].shape)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        # self.num_heads = [1, 1]
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            """
                # Machine node embedding
                self.get_machines = nn.ModuleList()
                # (in_size_ope=6, in_size_ma=3), out_size_ma=8, num_heads=[1, 1], dropout=0.1
                self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                                 self.dropout, self.dropout, activation=F.elu))
                for i in range(1, len(self.num_heads)):
                    # (out_size_ope=8, out_size_ma=8), out_size_ma=8, num_heads[1]=1, dropout=0.1, dropout=0.1
                    self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                                     self.dropout, self.dropout, activation=F.elu))
                h_mas = attention embedding of machine
            """
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, features)

            """features: (ope, mas, proc_time)"""
            features = (features[0], h_mas, features[2])
            # Second Stage, operation node embedding
            # shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, features)
            "embedding of operations"
            features = (h_opes, features[1], features[2])
            # print(f"shape ope, machine, time: {features[0].shape}, {features[1].shape}, {features[2].shape}")

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # 按照机器数目pool，得到shape: [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))  # 根据工序pool，得到batch和embedding
            h_opes_pooled = torch.stack(h_opes_pooled)  # shape: [len(batch_idxes), d]
        else:
            h_opes_pooled = h_opes.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ope]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        # biases是每个job的起始索引，其实是里面工序的起始， end_ope_biases是job内工序的最大索引
        # ope_step是工序索引吧
        # size: (batch_size, job_num)
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        # shape: (batch_size, job_num, 8)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]

        # h_opes.shape: (batch, num_opes, 8)
        # h_job.shape:(batch_size, job_num, 8)
        h_jobs = h_opes.gather(1, jobs_gather)

        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                                                                   ope_step_batch[..., :, None].expand(-1, -1,
                                                                                                       state.ope_ma_adj_batch.size(
                                                                                                           -1))[
                                                                       batch_idxes])
        """
        h_job.shape:(batch_size, job_num, 8)
        h_jobs_padding.shape: (batch_size, job_num, mas_num, 8)
        h_mas_padding.shape: (batch_size, job_num, mas_num, 8)
        h_mas_pooled_padding.shape: (batch_size, job_num, mas_num, 8)
        h_opes_pooled_padding.shape: (batch_size, job_num, mas_num, 8)
        """
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return

        # ====================================
        # ==========  计算Action     ==========
        # ====================================
        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
        # print("-----------shape------------")
        # print("mask score", eligible.shape)
        mask = eligible.transpose(1, 2).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)
        # print("flatten:", mask.shape, scores.shape)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        # print("probs:", action_probs.shape)
        # print("action_probs shape:", action_probs.shape)
        # Store data in memory during training
        if flag_train == True:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch))
            memories.ope_pre_adj.append(copy.deepcopy(state.ope_pre_adj_batch))
            memories.ope_sub_adj.append(copy.deepcopy(state.ope_sub_adj_batch))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

        return action_probs, ope_step_batch, h_pooled

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 jobs_gather, eligible, action_envs, flag_sample=False):
        """

        :param ope_ma_adj:
        :param ope_pre_adj:
        :param ope_sub_adj:
        :param raw_opes:
        :param raw_mas:
        :param proc_time:
        :param jobs_gather:
        :param eligible:
        :param action_envs:
        :param flag_sample:
        :return:
        """
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        # L iterations of the HGNN
        h_mas, h_opes = None, None
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)
            features = (h_opes, features[1], features[2])
        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_opes.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_jobs = h_opes.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys


class PPO(object):
    def __init__(self, train_paras):
        self.policy = HGNNScheduler()
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating
        # print("------------------------")
        # print(len(memory.ope_ma_adj))  # 48
        # print("------------------------")
        # print(torch.stack(memory.ope_ma_adj, dim=0).shape)  # [48, 20, 48, 5]
        # print("------------------------")
        # print(torch.stack(memory.ope_ma_adj, dim=0).transpose(0, 1).shape)  # [20, 48, 48, 5]
        # print("------------------------")
        # print(torch.stack(memory.ope_ma_adj, dim=0).transpose(0, 1).flatten(0,1).shape)  # [960, 48, 5]
        # print("------------------------")
        # Flatten the data in memory (in the dim of parallel instances and decision points)
        s_dim = 0
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_ope_sub_adj = torch.stack(memory.ope_sub_adj, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=s_dim).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=s_dim)  # .transpose(0, 1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=s_dim).transpose(0, 1)
        old_logprobs = torch.stack(memory.logprobs, dim=s_dim).transpose(0, 1).flatten(0, 1)
        old_action_envs = torch.stack(memory.action_indexes, dim=s_dim).transpose(0, 1).flatten(0, 1)
        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[start_idx: end_idx, :, :],
                                         old_ope_pre_adj[start_idx: end_idx, :, :],
                                         old_ope_sub_adj[start_idx: end_idx, :, :],
                                         old_raw_opes[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_proc_time[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_envs[start_idx: end_idx])


@dataclass
class EnvState:
    '''
        Class for the state of the environment
        '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None
    idle_times_batch: torch.Tensor = None
    # dynamic
    """长度为batch的一维tensor"""
    batch_idxes: torch.Tensor = None
    """ feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"] 6, self.num_opes)) """
    feat_opes_batch: torch.Tensor = None
    """ feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"] 3, num_mas)) """
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None
    ope_ma_order_matrix: torch.Tensor = None


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    feat_job_batch， 每个job拥有多少个机器
    param1: 每个工序的所属job
    在维度1取索引，返回
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)


class FJSPEnv(object):
    def __init__(self, num_mas=10, num_jobs=20, num_opes=50, batch_size=100):
        self.paras = {"ope_feat_?dim": 6, "ma_feat_dim": 3}
        self.batch_size = batch_size
        self.num_mas = num_mas
        self.num_jobs=num_jobs
        self.num_opes = num_opes
        # dynamic feats
        # shape: (batch_size, num_opes, num_mas) 工序和机器分配表
        self.proc_times_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_mas))
        # shape: (batch_size, num_opes, num_mas) 工序的邻域机器，所有可分配的机器表格
        self.ope_ma_adj_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_mas))
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_mas))

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_opes))
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_opes))
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.randint(0, 10, size=(batch_size, num_opes))
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.randint(0, 10, size=(batch_size, num_jobs))
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.randint(0, 10, size=(batch_size, num_jobs))
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = torch.randint(0, 10, size=(batch_size, num_jobs))
        # shape: (batch_size), 工序数：the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(batch_size)  # Uncompleted instances
        self.time = torch.zeros(batch_size)  # Current time of the environment
        self.N = torch.zeros(batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        # biases, 也就是每个工序的开始索引
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.feat_opes_batch = None
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(batch_size, self.paras["ope_feat_dim"], num_opes))
        feat_mas_batch = torch.zeros(size=(batch_size, self.paras["ma_feat_dim"], num_mas))
        # 论文里的特征2：邻域机器结点数：可选机器数
        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        # 论文里的特征3：在机器上的平均处理时间：处理时间求和然后除以邻域机器数（未分配）
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        # 论文里的特征5：尚未调度的工序，初始化为每个工序所属job的工序数[num1, num1, ..., numJ]
        # return feat_job_batch.gather(1, opes_appertain_batch)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        # 论文里的特征？？：开始时间？？批量乘，batch相等就是矩阵乘。通过选择矩阵乘得到初始化的完工时间，即平均执行时间
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        # 论文里的特征6：Job结束时间，即makespan
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        # Machine的特征1：邻域工序数
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)

        self.feat_opes_batch = feat_opes_batch
        '''
        features, dynamic
            ope:
                0 Status
                1 Number of neighboring machines
                2 Processing time
                3 Number of unscheduled operations in the job
                4 Job completion time
                5 Start time
            ma:
                0 Number of neighboring operations
                1 Available time
                2 Utilization
        '''

        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # 初始化为都为处于执行状态，即False
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)
        # 工序的动态调度状态：
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]

        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)


if __name__ == '__main__':
    # fjsp_env = FJSPEnv()
    # state = fjsp_env.state
    # done = False
    # dones = fjsp_env.done_batch
    # model = PPO(train_paras={"K_epochs": 3})
    HGNN = HGNNScheduler()
    print(HGNN)

    # shape: (batch_size, num_opes, num_mas) 工序的邻域机器，所有可分配的机器表格
    batch_size, num_opes, num_mas = 100, 6, 3
    ope_feat_dim, ma_feat_dim = 6, 3
    batch_idxes = torch.arange(batch_size).long()  # Uncompleted instances
    # flag_sample = False
    # flag_train = False

    feat_opes_batch = torch.randint(0, 10, size=(batch_size, ope_feat_dim, num_opes)).float()
    feat_mas_batch = torch.randint(0, 10, size=(batch_size, ma_feat_dim, num_mas)).float()
    # shape: (batch_size, num_opes, num_mas) 工序和机器分配表
    proc_times_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_mas)).float()
    features = (feat_opes_batch, feat_mas_batch, proc_times_batch)

    ope_ma_adj_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_mas)).long()
    # shape: (batch_size, num_opes, num_opes)
    ope_pre_adj_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_opes)).long()
    # shape: (batch_size, num_opes, num_opes)
    ope_sub_adj_batch = torch.randint(0, 10, size=(batch_size, num_opes, num_opes)).long()

    # self.num_heads = [1, 1]
    for i in range(len([1, 1])):
        h_mas = HGNN.get_machines[i](ope_ma_adj_batch=ope_ma_adj_batch,
                                     batch_idxes=batch_idxes,
                                     feat=features)
        print(h_mas.shape)
        """features: (ope, mas, proc_time)"""
        features = (features[0], h_mas, features[2])
        # Second Stage, operation node embedding
        # shape: [len(batch_idxes), max(num_opes), out_size_ope]
        h_opes = HGNN.get_operations[i](ope_ma_adj_batch=ope_ma_adj_batch,
                                        ope_pre_adj_batch=ope_pre_adj_batch,
                                        ope_sub_adj_batch=ope_sub_adj_batch,
                                        batch_idxes=batch_idxes,
                                        feats=features)
        "embedding of operations"
        features = (h_opes, features[1], features[2])
        print(f"shape ope, machine, time: {features[0].shape}, {features[1].shape}, {features[2].shape}")

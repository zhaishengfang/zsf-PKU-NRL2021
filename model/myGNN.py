# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.nn import global_max_pool as gap
from torch_geometric.nn import GraphConv


class GCN_cls_e(torch.nn.Module):
    def __init__(self, hidden_channels, Num_node_features, num_classes):
        super(GCN_cls_e, self).__init__()
        torch.manual_seed(12345)
        Conv = GraphConv
        self.bond_dim = 10
        self.bond_embed = nn.Linear(self.bond_dim, 1)
        self.conv1 = Conv(Num_node_features, hidden_channels)
        self.conv2 = Conv(hidden_channels, hidden_channels)
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        # self.sig = nn.Sigmoid()
    
    def forward(self, x, edge_index, edge_a, batch):
        edge_a = self.bond_embed(edge_a)
        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index, edge_a)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_a)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        # 2. Readout layer
        x = x1 + x2
        x = self.lin1(x)
        x = x.relu()
        
        #
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x  # .squeeze(-1)  # self.sig(x)


class GCN_reg_e(torch.nn.Module):
    def __init__(self, hidden_channels, Num_node_features):
        super(GCN_reg_e, self).__init__()
        torch.manual_seed(42)
        Conv = GraphConv
        self.bond_dim = 10
        self.bond_embed = nn.Linear(self.bond_dim, 1)
        self.conv1 = Conv(Num_node_features, hidden_channels)
        self.conv2 = Conv(hidden_channels, hidden_channels)
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, edge_a, batch):
        # 进行边特征嵌入
        edge_a = self.bond_embed(edge_a)
        x = self.conv1(x, edge_index, edge_a)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_a)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        # Readout layer
        x = x1 + x2
        x = self.lin1(x)
        x = x.relu()
        
        # output
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x.squeeze(-1)  # self.sig(x)

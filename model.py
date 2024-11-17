import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GATv2Conv


class GCNSample(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != self.layers.__len__()-1:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.g = g
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads, activation=F.elu)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, num_heads)
        self.droupout = nn.Dropout(0.2)

    def forward(self, g, features):
        h = self.layer1(g, features).flatten(1)
        h = self.droupout(h)
        h = self.layer2(g, h).mean(1)
        return h


class GATv2(nn.Module):
    # class dgl.nn.pytorch.conv.GATv2Conv(
    # in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, share_weights=False)
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GATv2, self).__init__()
        self.layer1 = GATv2Conv(
            in_dim, hidden_dim, num_heads, activation=F.elu)
        self.layer2 = GATv2Conv(hidden_dim * num_heads, out_dim, num_heads=1)
        self.droupout = nn.Dropout(0.2)

    def forward(self, g, features):
        h = self.layer1(g, features).flatten(1)
        h = self.droupout(h)
        h = self.layer2(g, h).mean(1)
        return h


class SAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, aggregator_type):
        super(SAGE, self).__init__()
        self.layer1 = SAGEConv(in_dim, hidden_dim, aggregator_type)
        self.layer2 = SAGEConv(hidden_dim, out_dim, aggregator_type)
        self.droupout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = F.relu(h)
        h = self.droupout(h)
        h = self.layer2(g, h)
        return h


class SAGEOneLayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator_type):
        super(SAGEOneLayer, self).__init__()
        self.layer1 = SAGEConv(
            in_dim, out_dim, aggregator_type, activation=F.relu)

    def forward(self, g, features):
        h = self.layer1(g, features)
        return h

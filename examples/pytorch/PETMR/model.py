from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from VGAE_train import device


class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        print('hid_dim1 $ hid_dim2 = ',hidden2_dim,hidden1_dim)

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)
        self.pool = SumPooling()
        self.dense = nn.Linear(64, 1)

    def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        print('This is the encoding feature', z.shape)
        hg = self.pool(g, z)
        hg = torch.tanh(hg)
        print('This is the pooling feature', hg.shape)
        # print('This is the dense feature', self.dense(hg).shape)
        adj_rec = self.decoder(z)
        return adj_rec


class GCN(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, 64)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 64)
        self.pool = SumPooling()
        self.dense = nn.Linear(64, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        hg = self.pool(g, h)
        hg = torch.tanh(hg)
        return self.dense(hg)
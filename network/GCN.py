import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Parameter

class GraphConvolution(nn.Module):
    """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input.float(), self.weight.float())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# Create adjacency matrix from statistics.
def gen_A(num_classes, t, p, adj_data):
    adj = np.array(adj_data['adj']).astype(np.float32)
    nums = np.array(adj_data['nums']).astype(np.float32)
    nums = nums[:, np.newaxis]
    adj = adj / nums
    adj[adj < t] = 0
    adj[adj >= t] = 1
    adj = adj * p / (adj.sum(0, keepdims=True) + 1e-6)
    adj = adj + np.identity(num_classes, np.int_)
    return adj

# Apply adjacency matrix re-normalization.
def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D).type_as(A)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#










































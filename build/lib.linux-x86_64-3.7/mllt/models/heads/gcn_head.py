import torchvision.models as models
from torch.nn import Parameter
from torch.autograd import Variable
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
import pickle

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False,dropout_n = 0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout_n = dropout_n
        self.dropout = nn.Dropout(self.dropout_n) 
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

@HEADS.register_module
class GcnHead(nn.Module):
    def __init__(self,
                 in_channels=512,
                 num_classes=80,
                 word2vec_len=300,
                 mid_channels=1024,
                 t=0,
                 ave_pool=False,
                 adj_file='mllt/appendix/coco_adj.pkl',
                 word2vec_file='mllt/appendix/coco_glove_word2vec.pkl',
                 loss_cls = dict(
                 type='CrossEntropyLoss',
                 use_sigmoid=False,
                 loss_weight=1.0)):
        super(GcnHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ave_pool=ave_pool
        self.loss_cls = build_loss(loss_cls)

        self.A = self.gen_adj(t,adj_file)
        self.word2vec = self.gen_word2vec(word2vec_file)

        self.relu = nn.LeakyReLU(0.2)
        self.gc1 = GraphConvolution(word2vec_len, mid_channels)
        self.gc2 = GraphConvolution(mid_channels,self.in_channels)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def init_weights(self):
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        if self.ave_pool:
            x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        self.adj = gen_adj(self.A).detach()

        g = self.gc1(self.word2vec, self.adj)
        g = self.relu(g)
        g = self.gc2(g, self.adj)

        g = g.transpose(0, 1)
        cls_score = torch.matmul(x, g)

        return cls_score


    def loss(self,
             cls_score,
             labels,
             reduction_override=None):
        losses = dict()
        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels,
            None,
            avg_factor=None,
            reduction_override=reduction_override)
        losses['acc'] = accuracy(cls_score, labels)
        return losses

    def gen_adj(self, t, adj_file):
        _adj = gen_A(self.num_classes, t, adj_file)
        A = Parameter(torch.from_numpy(_adj).float())
        return A

    def gen_word2vec(self, inp_name):
        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                word2vec = pickle.load(f)
        else:
            word2vec = np.identity(self.num_classes)  # is that okay? shouldn't it be 300*80?
        word2vec = Variable(torch.from_numpy(word2vec).float()).cuda()
        return word2vec

# def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=300, which_model='resnet101'):
#
#     model = models.resnet101(pretrained=pretrained)
#     print("Loading pretrained res101 models")
#
#     return GCNResnet(model, num_classes, t=t, adj_file=adj_file, word2vec_len=in_channel, which_model=which_model)


# utils
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

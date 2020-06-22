import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
import math
from torch.nn.parameter import Parameter

@HEADS.register_module
class CosHead(nn.Module):
    """Simplest classification head, with only one fc layer for classification"""

    def __init__(self,
                 in_channels=256,
                 num_classes=80,
                 margin=0.5,
                 in_scale = 1,
                 out_scale = 1.0,
                 bias = False,
                 squash=False,
                 init_std=0.001,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)):
        super(CosHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.margin = margin
        self.squash = squash
        self.bias = bias
        self.weight = Parameter(torch.Tensor(num_classes, in_channels).cuda())
        self.in_scale = in_scale
        self.out_scale = nn.Parameter(torch.FloatTensor(1).fill_(out_scale).cuda(),
                                      requires_grad=True) if out_scale > 0 else 1.
        if self.bias:
            self.cos_bias = Parameter(torch.Tensor(num_classes).fill_(1).cuda(), requires_grad=True)


    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        norm_x = torch.norm(x.clone(), 2, 1, keepdim=True)
        if self.squash:
            ex = (norm_x / (1 + norm_x)) * (x / norm_x)
        else:
            ex = x / norm_x
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        cls_score = self.out_scale * torch.mm(self.in_scale * ex, ew.t())
        if self.bias:
            cls_score = cls_score + self.cos_bias
        return cls_score

    def loss(self,
             cls_score,
             labels,
             weight=None,
             reduction_override=None):
        losses = dict()
        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels,
            weight,
            avg_factor=None,
            reduction_override=reduction_override)
        losses['acc'] = accuracy(cls_score, labels)
        return losses

'''
    def nonzero_cosine_similarity(self, x1, x2):
        # mask = x1 != 0
        cls_score = torch.zeros(x1.shape[0], self.num_classes).cuda()
        for i, x in enumerate(x1):
            mask = (x != 0).float()
            # print(torch.sum(mask))
            cls_score[i] = torch.cosine_similarity(x.unsqueeze(0),x2*mask)

        return cls_score
'''

if __name__ == '__main__':
    pass
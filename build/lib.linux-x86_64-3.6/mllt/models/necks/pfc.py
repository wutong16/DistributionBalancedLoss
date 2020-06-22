from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
from ..registry import NECKS


@NECKS.register_module
class PFC(nn.Module):
   
    def __init__(self, in_channels, out_channels, dropout, norm=False,relu=False,layers=1):
        super(PFC, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.fc_hidden_1 = nn.Linear(out_channels, out_channels)
        self.norm = norm
        self.dropout = dropout
        self.layers = layers
        self.avg_pool = nn.AvgPool2d(in_channels)
        self.relu= nn.LeakyReLU(0.2, inplace=True) if relu else None

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
    
    def init_weights(self):
        init.kaiming_normal_(self.fc.weight, mode='fan_out')
        init.constant_(self.fc.bias, 0)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = F.avg_pool2d(x, x.size()[2:])
        # y = self.avg_pool(x)
        # assert torch.max(abs(x-y)) == 0, print(x[1], y[1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.layers == 2:
            x = self.fc_hidden_1(x)
            x = self.bn(x)
            if self.dropout > 0:
                x = self.drop(x)

        if self.relu is not None:
            x = self.relu(x)
        return x


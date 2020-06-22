import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=0,
                 end_level=-1,
                 embedding=True,
                 dropout=0,
                 norm=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.activation = activation
        self.embedding = embedding

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
        self.start_level = start_level
        self.end_level = end_level
        self.num_pyramid = self.backbone_end_level - self.start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        if self.embedding:
            self.embedding_conv = ConvModule(
                out_channels*self.num_pyramid,
                out_channels*self.num_pyramid,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.fc = nn.Linear(out_channels*self.num_pyramid, out_channels)
            self.bn = nn.BatchNorm1d(out_channels)
            self.dropout = dropout
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            self.norm = norm

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = []
        for i in range(used_backbone_levels):
            if i == 0:
                outs.append(self.fpn_convs[i](laterals[i]))
            else:
                outs.append(
                    F.interpolate(self.fpn_convs[i](laterals[i]), scale_factor=2**i, mode='nearest'))
                    
        out = torch.cat(outs, dim=1)
        
        if self.embedding:
            out = self.embedding_conv(out)
            out = F.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out = self.bn(out)
            if self.norm:
                out = F.normalize(out)
            if self.dropout > 0:
                out = self.drop(out)

        return out

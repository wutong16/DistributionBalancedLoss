import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Function

from mmcv.cnn import constant_init, kaiming_init

from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS


class WeldonPool2dFunction(Function):

    def __init__(self, kmax, kmin):
        super(WeldonPool2dFunction, self).__init__()
        self.kmax = kmax
        self.kmin = kmin

    def get_number_of_instances(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input):
        # get batch information
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        # get number of regions
        n = h * w

        # get the number of max and min instances
        kmax = self.get_number_of_instances(self.kmax, n)
        kmin = self.get_number_of_instances(self.kmin, n)

        # sort scores
        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

        # compute scores for max instances
        self.indices_max = indices.narrow(2, 0, kmax)
        output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

        if kmin > 0:
            # compute scores for min instances
            self.indices_min = indices.narrow(2, n-kmin, kmin)
            output.add_(sorted.narrow(2, n-kmin, kmin).sum(2).div_(kmin)).div_(2)

        # save input for backward
        self.save_for_backward(input)
        # return output with right size
        return output.view(batch_size, num_channels)

    def backward(self, grad_output):

        # get the input
        input, = self.saved_tensors

        # get batch information
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        # get number of regions
        n = h * w

        # get the number of max and min instances
        kmax = self.get_number_of_instances(self.kmax, n)
        kmin = self.get_number_of_instances(self.kmin, n)

        # compute gradient for max instances
        grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)
        grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, self.indices_max, grad_output_max).div_(kmax)

        if kmin > 0:
            # compute gradient for min instances
            grad_output_min = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, self.indices_min, grad_output_min).div_(kmin)
            grad_input.add_(grad_input_min).div_(2)

        return grad_input.view(batch_size, num_channels, h, w)


class WeldonPool2d(nn.Module):

    def __init__(self, kmax=1, kmin=None):
        super(WeldonPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax

    def forward(self, input):
        return WeldonPool2dFunction(self.kmax, self.kmin)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ')'



@HEADS.register_module
class WeldonHead(nn.Module):
    """ Weldon classification head,
        https://ieeexplore.ieee.org/abstract/document/8242666
    """

    def __init__(self,
                 in_channels=2048,
                 num_classes=80,
                 kmax=1,
                 kmin=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)):
        super(WeldonHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.transfer = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.spatial_pooling = WeldonPool2d(kmax, kmin)
       
        self.debug_imgs = None

    def init_weights(self):
        kaiming_init(self.transfer)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = self.transfer(x)
        x = self.spatial_pooling(x)
        cls_score = x.view(x.size(0), -1)
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

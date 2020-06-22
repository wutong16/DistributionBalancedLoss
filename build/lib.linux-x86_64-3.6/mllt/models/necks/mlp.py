from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from ..registry import NECKS


@NECKS.register_module
class MLP(nn.Module):
   
    def __init__(self, in_channels, bottle_neck, out_channels, dropout1, dropout2, norm=False):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_channels, bottle_neck)
        self.bn1 = nn.BatchNorm1d(bottle_neck)
        self.fc2 = nn.Linear(bottle_neck, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.norm = norm
        if self.dropout1 > 0:
            self.drop1 = nn.Dropout(self.dropout1)
        if self.dropout2 > 0:
            self.drop2 = nn.Dropout(self.dropout2)
    
    def init_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.bn1.weight, 1)
        init.constant_(self.bn1.bias, 0)
        init.kaiming_normal_(self.fc2.weight, mode='fan_out')
        init.constant_(self.fc2.bias, 0)
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.dropout1 > 0:
            x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        if self.norm:
            x = F.normalize(x)
        if self.dropout2 > 0:
            x = self.drop2(x)
        return x

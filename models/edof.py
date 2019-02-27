from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class EdofNet(nn.Module):

    def hidden_block(self, dilation):
        return nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(32))

    def __init__(self, max_dilation,device):
        super(EdofNet, self).__init__()
        self._stride = 1
        self.input_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32))
        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1))

        self.encode = np.arange(2, max_dilation+1)
        self.decode = np.flip(np.arange(2,max_dilation))
        self.dilations = np.append(self.encode, self.decode)

        self.hidden_blocks = nn.ModuleList()
        for j,i in enumerate(self.dilations):
            self.hidden_blocks.append(self.hidden_block(i))
            self.hidden_blocks[j].to(device)

    def forward(self, x):
        y = self.input_layer(x)
        for block in self.hidden_blocks:
            y = F.relu(block(y))
        y = self.output_layer(y)
        y = y + x
        return y


if __name__ == '__main__':
    net = EdofNet(max_dilation=5)
    a = torch.zeros((1,3,32,32))
    b = net(a)
    print ("Done")


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def psi_to_depth(psi, focal_point=1.1, params=None):
    Zn = focal_point
    Lambda = params['Lambda'] if params else (455*1e-9)
    D = params['D'] if params else (2.28*1e-3)
    r = D / 2
    pi = math.pi
    Zo = (1 / ((1/Zn) + ((psi * Lambda) / (pi * r**2))))
    return Zo

class Dfd_net(nn.Module):
    def __init__(self, target_mode=None, num_class=16, mode='classification',skip_layer=True,pool=True):
        super(Dfd_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(16, momentum=0.99)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5,padding=2)
        self.batch_norm2 = nn.BatchNorm2d(36, momentum=0.99)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(36, 64, 5,padding=2)
        self.batch_norm3 = nn.BatchNorm2d(64, momentum=0.99)
        self.pool3 = nn.MaxPool2d(2, 2,ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, 3,padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128, momentum=0.99)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128,256,3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256, momentum=0.99)
        self.pool5 = nn.MaxPool2d(2,2)
        self.conv6 = nn.Conv2d(256, num_class, 1)
        self.dense = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, num_class)
        self.upsampling = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class,kernel_size=64, stride=32)
        self.conv7_from_pool4 = nn.Conv2d(128, num_class, kernel_size=1)
        self.upsampling16 = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class, kernel_size=32,stride=16)
        self.conv9_from_pool3 = nn.Conv2d(64, num_class, kernel_size=1)
        self.upsampling8 = nn.ConvTranspose2d(in_channels=num_class, out_channels=num_class, kernel_size=16, stride=8)
        self.conv8_reg = nn.Conv2d(16,1,1,1)
        self.conv8_reg.weight.data[0,:,0,0] = torch.arange(16).float()
        self.softmax = nn.Softmax()
        self.num_class = num_class
        self.mode = mode
        self.skip_layer = skip_layer
        self.target_mode = target_mode
        self.pool = pool


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool3(x)
        skip8 = x
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool4(x)
        skip = x
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.pool5(x)
        x = self.conv6(x)
        if self.mode == 'segmentation':
            x = self.upsampling(x)
            if self.skip_layer:
                skip_upsampled = self.upsampling16(self.conv7_from_pool4(skip))[:,:,8:-8,8:-8]
                skip_upsampled8 = self.upsampling8(self.conv9_from_pool3(skip8))[:,:,4:-4,4:-4]
                x = x[:,:,16:-16,16:-16]
                x = x + skip_upsampled + skip_upsampled8
            if self.target_mode == 'cont':
                x = self.softmax(x)
                x = self.conv8_reg(x)
                if self.pool:
                    x = F.avg_pool2d(x, 4)
                    x = psi_to_depth(x)
                else:
                    x = torch.squeeze(x)
        else:
            x = x.view(-1, self.num_class)
            # x = self.upsampling8(x)[:,:,4:-4,4:-4]
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfFuse(nn.Module):
    def __init__(self):
        super(ConfFuse, self).__init__()
        self.conv_L_mono = nn.Conv2d(16,1,1)
        self.conv_R_mono = nn.Conv2d(16,1,1)
        self.conv_stereo = nn.Conv2d(192, 1, 1)
        self.conv8_reg = nn.Conv2d(2, 1, 1, 1, bias=False)
        self.conv8_reg.weight.data[0, :, 0, 0] = torch.Tensor([1,1]).float()
        for param in self.conv8_reg.parameters():
            param.requires_grad = False

    def forward(self, mono_L, mono_R, stereo, mono_L_conf, mono_R_conf, stereo_conf):
        mono_L_conf = self.conv_L_mono(mono_L_conf)
        mono_R_conf = self.conv_R_mono(mono_R_conf)
        stereo_conf = self.conv_stereo(stereo_conf)
        conf = torch.cat((mono_L_conf, mono_R_conf, stereo_conf), 1)
        conf = nn.Softmax(dim=1)(conf)
        conf = conf[:,0,:,:] * mono_L + conf[:,1,:,:] * mono_R + conf[:,2,:,:] * stereo

        return conf

class MaskFuse(nn.Module):
    def __init__(self, H, W):
        super(MaskFuse, self).__init__()
        # self.mask = nn.Parameter(torch.round(torch.randn((H,W)).type(torch.float)), requires_grad=True)
        self.low_th = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.high_th = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.conv = nn.Conv2d(2,1,1)
        # self.mask = Variable(torch.zeros((512,1024)), requires_grad=True)
        # self.mask = nn.init.normal_(self.mask, mean=0.5, std=0.5)

    def get_input_in_ranges(self, input, low_th=None, high_th=None):
        if low_th:
            input = F.relu(input - low_th) + low_th
        if high_th:
            input = -(F.relu(-input+high_th) - high_th)
        return input


    def forward(self, mono,stereo):
        mono = self.get_input_in_ranges(mono, self.low_th, self.high_th)
        mono = torch.unsqueeze(mono,1)
        stereo = self.get_input_in_ranges(stereo, self.high_th, None)
        stereo = torch.unsqueeze(stereo, 1)
        fuse = torch.cat((mono, stereo),1)
        fuse = self.conv(fuse)
        fuse = torch.squeeze(fuse,1)
        return fuse



class UNet(nn.Module):
    def __init__(self, in_channels=2, n_classes=128 ,depth=5, wf=6, padding=True,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.regres1 = nn.Conv2d(128,1,1)
        # self.regres2 = nn.Conv2d(32,8,1)
        # self.regres3 = nn.Conv2d(8,1,1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        x = self.last(x)
        x = nn.Softmax()(x)
        x = self.regres1(x)
        # x = self.regres2(x)
        # x = self.regres3(x)

        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out
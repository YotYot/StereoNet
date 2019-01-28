from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *
from dfd import Dfd_net

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp, dfd_net = True, dfd_at_end=True, device=None):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.dfd_net = dfd_net
        self.dfd_at_end = dfd_at_end

        if self.dfd_net and not self.dfd_at_end:
            features = 66
        else:
            features = 64

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(features, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        self.last_conv = nn.Conv2d(2,1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # self.last_conv.weight.data = torch.Tensor([[[[0.5]],[[0.5]]]])

        if self.dfd_net:
            if self.dfd_at_end:
                self.dfd = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
            else:
                self.dfd = Dfd_net(mode='segmentation', target_mode='cont', pool=True)
            self.dfd.to(device)
            model_path = 'models/dfd_checkpoint/checkpoint_274.pth.tar'
            print("loading checkpoint for dfd net")
            checkpoint = torch.load(model_path, map_location=device)
            self.dfd.load_state_dict(checkpoint['state_dict'], strict=True)
            #Freeze the net if at features level
            if not self.dfd_at_end:
                print("Freezing dfd net")
                for child in self.dfd.children():
                    for param in child.parameters():
                        param.requires_grad = False


    def forward(self, left, right):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        if self.dfd_net:
            dfd_left_out = self.dfd(left)
            dfd_right_out = self.dfd(right)
            import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            # plt.imshow(np.transpose(((left[0]+1)/2).detach().cpu().numpy(), (1,2,0)))
            # plt.subplot(2, 1, 2)
            # plt.imshow(dfd_left_out[0].cpu())
            # plt.show()


            if not self.dfd_at_end:
                refimg_fea = torch.cat((refimg_fea, dfd_left_out), dim=1)
                targetimg_fea = torch.cat((targetimg_fea, dfd_right_out), dim=1)
        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp/4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp/4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)
        # pred3 = torch.unsqueeze(pred3,dim=1)
        # dfd_left_out = F.upsample(dfd_left_out, scale_factor=4)
        # pred4 = torch.cat((pred3, dfd_left_out), dim=1)
        # pred4 = self.last_conv(pred4)
        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3

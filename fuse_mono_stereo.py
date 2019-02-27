from __future__ import print_function
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from sintel_io import depth_read
from dataloader import sintel_listflowfile as lt
# from dataloader import sintel_listflowfile_with_filter as lt
# from dataloader import sintel_listflowfile_without_filter_with_depth as lt
from dataloader import SintelFlowLoader as DA
from models import *
import pickle
from dfd import Dfd_net, psi_to_depth
from edof import EdofNet
# from fuse_net import Fuse
# from fuse_net import UNet
from fuse_net import MaskFuse
from fuse_net import ConfFuse
from local_utils import load_model, depth2disparity
import matplotlib.pyplot as plt
from disparity_mapping import apply_disparity


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/',
                    help='datapath')
parser.add_argument('--left_imgs', default=None,
                    help='left img train dir name')
parser.add_argument('--right_imgs', default=None,
                    help='left img train dir name')
parser.add_argument('--disp_imgs', default=None,
                    help='left img train dir name')
parser.add_argument('--disp_R_imgs', default=None,
                    help='right img train dir name')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--focus_L', default='1500',
                    help='Left img focus point')
parser.add_argument('--focus_R', default='700',
                    help='Right img focus point')
# parser.add_argument('--loadmodel', default=None,
parser.add_argument('--loadmodel', default='./checkpoints/checkpoint_50.tar',
# parser.add_argument('--loadmodel', default='./checkpoints/checkpoint_filter_loss_2.6.tar',
# parser.add_argument('--loadmodel', default='./checkpoints/checkpoint_clean_from_scratch_loss_2.1.tar',
# parser.add_argument('--loadmodel', default='./pretrained_model_KITTI2015.tar',
                    help='load model')
parser.add_argument('--loadfuse', default=None,
                    help='load fuse model')
parser.add_argument('--savemodel', default='./checkpoints/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--clean', action='store_true', default=False,
                    help='random seed (default: 1)')
parser.add_argument('--cont', action='store_true', default=False,
                    help='random seed (default: 1)')
parser.add_argument('--dfd', action='store_true', default=False,
                    help='include dfd net')
parser.add_argument('--dfd_at_end', action='store_true', default=False,
                    help='include dfd net')
parser.add_argument('--right_head', action='store_true', default=False,
                    help='right disp branch')
parser.add_argument('--pred_occlusion', action='store_true', default=False,
                    help='pred occlusion or right depth')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# test_imgs  = ['City_R_0116_1100_maskImg.png', 'City_R_0092_1100_maskImg.png', 'City_R_0114_1100_maskImg.png', 'City_R_0190_1100_maskImg.png', 'City_R_0154_1100_maskImg.png', 'City_R_0066_1100_maskImg.png', 'City_R_0202_1100_maskImg.png', 'City_R_0204_1100_maskImg.png', 'City_R_0128_1100_maskImg.png', 'City_R_0042_1100_maskImg.png', 'City_R_0026_1100_maskImg.png', 'City_R_0058_1100_maskImg.png']

# if args.clean:
#     test_imgs = [img.replace('_1100_maskImg.png', '.tif') for img in test_imgs]

# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath, args.left_imgs, args.right_imgs, args.disp_imgs,filenames=test_imgs, clean=args.clean)
all_left_img, all_right_img, all_left_disp, all_R_disp, test_left_img, test_right_img, test_left_disp, test_R_disp = lt.dataloader(args.datapath, args.left_imgs, args.right_imgs, args.disp_imgs, args.disp_R_imgs, clean=args.clean, focus_L=args.focus_L, focus_R=args.focus_R)



TrainImgLoader = torch.utils.data.DataLoader(
    # DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True,dploader=DA.depth_loader),
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, all_R_disp, training=True, dploader=depth_read,cont=args.cont),
    batch_size=4, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    # DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, dploader=DA.depth_loader),
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, test_R_disp, training=False, dploader=depth_read, cont=args.cont),
    batch_size=2, shuffle=False, num_workers=4, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, device=device, dfd_net=args.dfd, dfd_at_end=args.dfd_at_end, right_head=args.right_head)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    print("Loading stereo net checkpoint: ", args.loadmodel)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

dfd_net_D5 = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
dfd_net_D5 = dfd_net_D5.eval()
dfd_net_D5 = dfd_net_D5.to(device)
load_model(dfd_net_D5, device, model_path='/home/yotamg/PycharmProjects/dfd/trained_models/Net_continuous_dn1500_D5/checkpoint_254.pth.tar')


dfd_net = Dfd_net(mode='segmentation', target_mode='cont', pool=False)
dfd_net = dfd_net.eval()
dfd_net = dfd_net.to(device)
load_model(dfd_net, device, model_path='/home/yotamg/PycharmProjects/dfd/trained_models/Net_continuous_dn1500/checkpoint_257.pth.tar')

# edof_net = EdofNet(max_dilation=4, device=device)
# edof_net = edof_net.eval()
# edof_net = edof_net.to(device)
# load_model(edof_net,device, model_path='/home/yotamg/PycharmProjects/EDOF/trained_models/EdofNet_after_imaging_fix/checkpoint_99.pth.tar')

# fuse_net = Fuse()
# fuse_net = UNet()
# fuse_net = MaskFuse(256,512)
fuse_net = ConfFuse()
fuse_net = fuse_net.train()
fuse_net = fuse_net.to(device)

if args.loadfuse is not None:
    state_dict = torch.load(args.loadfuse)
    fuse_net.load_state_dict(state_dict['state_dict'])
    print ("Loading fuse net checkpoint: ", args.loadfuse)

optimizer = optim.Adam(fuse_net.parameters(), lr=0.1, betas=(0.9, 0.999))

def train(imgL, imgR, disp_true, disp_true_R):
    optimizer.zero_grad()
    model.eval()
    fuse_net.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))

    if args.cuda:
        imgL, imgR,disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    # ---------
    mask = disp_true < 192
    # ----

    with torch.no_grad():
        mono_L, mono_L_conf = dfd_net_D5(imgL, float(args.focus_L)*1e-3, D=(2.28*1e-3))
        mono_R, mono_R_conf = dfd_net(imgR, float(args.focus_R)*1e-3, D=(5*1e-3))
        if len(mono_L.shape) == 2:
            mono_L = torch.unsqueeze(mono_L,0).to(device)
            mono_R = torch.unsqueeze(mono_R,0).to(device)
        # plt.subplot(211)
        # plt.imshow(((imgR[0] + 1)/2).permute(1,2,0))
        # plt.subplot(212)
        # plt.imshow(mono_R[0])
        # plt.show()
        if args.right_head:
            if args.pred_occlusion:
                output3, occ, stereo_conf = model(imgL, imgR)
            else:
                output3, output_R, stereo_conf = model(imgL, imgR)
        else:
            output3, stereo_conf = model(imgL, imgR)

    disp_from_stereo = depth2disparity(imgR, output3,device)


    mono_R = torch.unsqueeze(mono_R,1)
    mono_RtoL = apply_disparity(mono_R, -disp_from_stereo)
    mono_RtoL = torch.squeeze(mono_RtoL, 1)
    # mono_RtoL = mono_RtoL * torch.abs(occ-1)

    mono_RtoL_conf = apply_disparity(mono_R_conf, -disp_from_stereo)
    # occ = torch.unsqueeze(occ, 1).repeat(1,16,1,1)
    # mono_RtoL_conf = mono_RtoL_conf * torch.abs(occ-1)

    # plt.subplot(221)
    # plt.imshow(mono_RtoL[0],vmin=torch.min(mono_R), vmax= torch.max(mono_R))
    # plt.subplot(222)
    # plt.imshow(mono_R[0][0],vmin=torch.min(mono_R), vmax= torch.max(mono_R))
    # plt.subplot(223)
    # plt.imshow((imgL[0].permute(1,2,0) + 1) / 2)
    # plt.subplot(224)
    # plt.imshow((imgR[0].permute(1,2,0) + 1) / 2)
    # plt.subplot(223)
    # plt.imshow(output3[0])
    # plt.subplot(224)
    # plt.imshow(disp_from_stereo[0])
    plt.show()


    # fuse_output = fuse_net(fuse_input)
    fuse_output = fuse_net(mono_L, mono_RtoL, output3, mono_L_conf, mono_RtoL_conf, stereo_conf)
    fuse_output = torch.squeeze(fuse_output, dim=1)
    # fuse_output = torch.squeeze(fuse_output, dim=3)
    loss = F.smooth_l1_loss(fuse_output, disp_true, size_average=True)
    loss.backward()
    optimizer.step()
    return loss.data

def test(imgL, imgR, disp_true):
    model.eval()
    dfd_net.eval()
    fuse_net.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()


    with torch.no_grad():
        mono_L, mono_L_conf = dfd_net(imgL, float(args.focus_L) * 1e-3, D=(5.0 * 1e-3))
        mono_R, mono_R_conf = dfd_net(imgR, float(args.focus_R) * 1e-3, D=(2.28 * 1e-3))
        if len(mono_L.shape) == 2:
            mono_L = torch.unsqueeze(mono_L,0)
            mono_R = torch.unsqueeze(mono_R,0)
        output3, occ, stereo_conf = model(imgL, imgR)

        disp_from_stereo = depth2disparity(imgR, output3,device)

        mono_R = torch.unsqueeze(mono_R, 1)
        mono_R = apply_disparity(mono_R, -disp_from_stereo)
        mono_R = torch.squeeze(mono_R, 1)
        # mono_R = mono_R * torch.abs(occ - 1)

        mono_R_conf = apply_disparity(mono_R_conf, -disp_from_stereo)
        occ = torch.unsqueeze(occ, 1).repeat(1, 16, 1, 1)
        # mono_R_conf = mono_R_conf * torch.abs(occ - 1)

        fuse_output = fuse_net(mono_L, mono_R, output3, mono_L_conf, mono_R_conf, stereo_conf)
        # try:
        #     fuse_output = torch.squeeze(fuse_output, dim=3)
        # except:
        #     fuse_output = fuse_output.permute(2,0,1)
    loss = torch.mean(torch.abs(fuse_output - disp_true))  # end-point-error
    rel_loss = torch.mean(torch.abs(fuse_output - disp_true) / disp_true)  # end-point-error

    mask_psi_range = (disp_true > 1.018) & (disp_true < 1.5)
    mask_psi1500_range_from_stereo = (output3 > 1.63) & (output3 < 1.83)
    # mask_psi1500_range_from_stereo = (disp_true > 1.34) & (disp_true < 1.59)
    mask_psi700_range_from_stereo = (output3 > 0.393) & (output3 < 1.018)
    # mask_psi700_range_from_stereo = (disp_true > 0.393) & (disp_true < 1.018)
    output_threshold_fuse = output3.clone()
    output_threshold_fuse[mask_psi1500_range_from_stereo] = mono_R[mask_psi1500_range_from_stereo]
    output_threshold_fuse[mask_psi700_range_from_stereo]  = mono_L[mask_psi700_range_from_stereo]


    mono_rel_loss = torch.mean(torch.abs(mono_L - disp_true) / disp_true)  # end-point-error
    mono_masked_rel_loss = 0
    mono700_masked_rel_loss = 0
    mono1500_masked_rel_loss = 0
    if len(mono_L[mask_psi_range]) != 0:
        mono_masked_rel_loss = torch.mean(torch.abs(mono_L[mask_psi_range] - disp_true[mask_psi_range]) / disp_true[mask_psi_range])  # end-point-error
    if len(mono_R[mask_psi700_range_from_stereo]) != 0:
        mono700_masked_rel_loss = torch.mean(torch.abs(mono_R[mask_psi700_range_from_stereo] - output3[mask_psi700_range_from_stereo]) / output3[mask_psi700_range_from_stereo])  # end-point-error
    if len(mono_L[mask_psi1500_range_from_stereo]) != 0:
        mono1500_masked_rel_loss = torch.mean(torch.abs(mono_L[mask_psi1500_range_from_stereo] - output3[mask_psi1500_range_from_stereo]) / output3[mask_psi1500_range_from_stereo])  # end-point-error
    stereo_rel_loss = torch.mean(torch.abs(output3 - disp_true) / disp_true)  # end-point-error
    threshold_fuse_rel_loss = torch.mean(torch.abs(output_threshold_fuse - disp_true) / disp_true)
    return loss, rel_loss, mono_rel_loss,mono_masked_rel_loss, stereo_rel_loss, threshold_fuse_rel_loss, mono700_masked_rel_loss, mono1500_masked_rel_loss

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001*(1/epoch)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()

    # ------------- TRAIN ------------------------------------------------------------
    total_train_loss = 0
    epochs = 300
    do_train = True
    for epoch in range(1,epochs):
        if do_train:
            for batch_idx, (imgL, imgR, disp_L, disp_R) in enumerate(TrainImgLoader):
                loss = train(imgL, imgR, disp_L, disp_R)
                print('Iter %d train loss = %.3f' % (batch_idx, loss))
                total_train_loss += loss

            if not os.path.isdir(args.savemodel):
                os.makedirs(args.savemodel)
            savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': fuse_net.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
            }, savefilename)
            print('total train loss = %.3f' % (total_train_loss / len(TrainImgLoader)))
            total_train_loss  = 0
        # ------------- TEST ------------------------------------------------------------
        if epoch % 1 == 0 or not do_train:
            total_test_loss = 0
            total_test_rel_loss = 0
            total_test_mono_rel_loss = 0
            total_test_mono_masked_rel_loss = 0
            total_test_mono_700_masked_rel_loss = 0
            total_test_mono_700_masked_rel_cnt = 0
            total_test_mono_1500_masked_rel_loss = 0
            total_test_mono_1500_masked_rel_cnt = 0
            total_test_stereo_rel_loss = 0
            total_test_mono_masked_rel_cnt = 0
            total_test_threshold_rel_loss = 0
            for batch_idx, (imgL, imgR, disp_L, disp_R) in enumerate(TestImgLoader):
                test_loss, rel_loss, mono_rel_loss, mono_masked_rel_loss, stereo_rel_loss, threshold_fuse_rel_loss, mono700_masked_rel_loss, mono1500_masked_rel_loss = test(imgL, imgR, disp_L)
                print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
                print('Iter %d test rel loss = %.3f' % (batch_idx, rel_loss))
                print('Iter %d test mono rel loss = %.3f' % (batch_idx, mono_rel_loss))
                print('Iter %d test mono masked rel loss = %.3f' % (batch_idx, mono_masked_rel_loss))
                print('Iter %d test mono 700 masked rel loss = %.3f' % (batch_idx, mono700_masked_rel_loss))
                print('Iter %d test mono 1500 masked rel loss = %.3f' % (batch_idx, mono1500_masked_rel_loss))
                print('Iter %d test stereo rel loss = %.3f' % (batch_idx, stereo_rel_loss))
                print('Iter %d test threshold rel loss = %.3f' % (batch_idx, threshold_fuse_rel_loss))
                # plt.show()
                total_test_loss += test_loss
                total_test_rel_loss += rel_loss
                total_test_mono_rel_loss += mono_rel_loss
                if mono_masked_rel_loss != 0:
                    total_test_mono_masked_rel_loss += mono_masked_rel_loss
                    total_test_mono_masked_rel_cnt += 1
                if mono700_masked_rel_loss != 0:
                    total_test_mono_700_masked_rel_loss += mono700_masked_rel_loss
                    total_test_mono_700_masked_rel_cnt += 1
                if mono1500_masked_rel_loss != 0:
                    total_test_mono_1500_masked_rel_loss += mono1500_masked_rel_loss
                    total_test_mono_1500_masked_rel_cnt += 1
                total_test_stereo_rel_loss += stereo_rel_loss
                total_test_threshold_rel_loss += threshold_fuse_rel_loss


            print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
            print('total test rel loss = %.3f' % (total_test_rel_loss / len(TestImgLoader)))
            print('total test mono rel loss = %.3f' % (total_test_mono_rel_loss / len(TestImgLoader)))
            print('total test mono masked rel loss = %.3f' % (total_test_mono_masked_rel_loss / total_test_mono_masked_rel_cnt))
            print('total test mono 700 masked rel loss = %.3f' % (total_test_mono_700_masked_rel_loss / total_test_mono_700_masked_rel_cnt))
            print('total test mono 1500 masked rel loss = %.3f' % (total_test_mono_1500_masked_rel_loss / total_test_mono_1500_masked_rel_cnt))
            print('total test stereo rel loss = %.3f' % (total_test_stereo_rel_loss / len(TestImgLoader)))
            print('total test threshold fuse rel loss = %.3f' % (total_test_threshold_rel_loss / len(TestImgLoader)))


if __name__ == '__main__':
    main()

from __future__ import print_function
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from local_utils import load_model
from disparity_mapping import apply_disparity
import matplotlib.pyplot as plt
import tqdm

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
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
parser.add_argument('--occ_L_dir', default='occ_flatten',
                    help='occlusions left dir')
parser.add_argument('--oof_L_dir', default='oof_flatten',
                    help='out-of-frame left dir')
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
all_left_img, all_right_img, all_left_disp, all_R_disp, all_L_occ, all_L_oof, test_left_img, test_right_img, test_left_disp, test_R_disp, test_left_occ, test_left_oof = lt.dataloader(
    args.datapath, args.left_imgs, args.right_imgs, args.disp_imgs, args.disp_R_imgs, args.occ_L_dir, args.oof_L_dir,
    clean=args.clean, focus_L=args.focus_L, focus_R=args.focus_R)

TrainImgLoader = torch.utils.data.DataLoader(
    # DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True,dploader=DA.depth_loader),
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, all_R_disp, all_L_occ, all_L_oof, True,
                     dploader=depth_read, cont=args.cont),
    batch_size=1, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    # DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, dploader=DA.depth_loader),
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, test_R_disp, test_left_occ, test_left_oof, False,
                     dploader=depth_read, cont=args.cont),
    batch_size=1, shuffle=False, num_workers=4, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, device=device, dfd_net=args.dfd, dfd_at_end=args.dfd_at_end,
                           right_head=args.right_head)
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

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

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

occlusion_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/occ_flatten'
oof_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Tau-agent/oof_flatten'

stereo_err_for_depth = dict()
mono_err_for_depth = dict()
mono_RtoL_err_for_depth = dict()
all_depth_cnt = dict()
stereo_occ_err_for_depth = dict()
mono_occ_err_for_depth = dict()
stereo_oof_err_for_depth = dict()
mono_oof_err_for_depth = dict()
occ_depth_cnt = dict()
oof_depth_cnt = dict()
for i in range(40000):
    stereo_err_for_depth[i] = 0
    mono_err_for_depth[i] = 0
    mono_RtoL_err_for_depth[i] = 0
    all_depth_cnt[i] = 0
    stereo_occ_err_for_depth[i] = 0
    mono_occ_err_for_depth[i] = 0
    occ_depth_cnt[i] = 0
    stereo_oof_err_for_depth[i] = 0
    mono_oof_err_for_depth[i] = 0
    oof_depth_cnt[i] = 0


def test(imgL, imgR, disp_true, disp_true_R, left_occ, left_oof):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    # ---------
    mask = disp_true < 192
    mask = torch.ones_like(mask)  # TODO - decide whether to mask anything
    # ----

    with torch.no_grad():
        mono_L,_ = dfd_net(imgL, int(args.focus_L)*1e-3, D=2.28*1e-3)
        mono_R,_ = dfd_net_D5(imgL, int(args.focus_R)*1e-3, D=5*1e-3)
        if args.right_head:
            if args.pred_occlusion:
                output_stereo, occ_L = model(imgL, imgR)
            else:
                output_stereo, output_R = model(imgL, imgR)
        else:
            output_stereo,_ = model(imgL, imgR)

    fuse_mask_min_1500 = output_stereo > 1.63
    fuse_mask_max_1500 = output_stereo < 1.83

    fuse_mask_min_700 = output_stereo > 0.393
    fuse_mask_max_700 = output_stereo < 1.017

    fuse_mask_1500 = fuse_mask_min_1500 & fuse_mask_max_1500
    fuse_mask_700  = fuse_mask_min_700 & fuse_mask_max_700

    fuse_mask_for_occ_oof = (disp_true > 0.3) & (disp_true < 1.2)

    occ_mask = fuse_mask_for_occ_oof & left_occ.byte().to(device)
    oof_mask = fuse_mask_for_occ_oof & left_oof.byte().to(device)
    occ_oof_mask = occ_mask | oof_mask

    if args.right_head:
        if args.pred_occlusion:
            occ_mask = torch.round(occ_L).byte() & fuse_mask_for_occ_oof
            oof_mask = [left_oof == 1][0].to(device)
            oof_mask = oof_mask & fuse_mask_for_occ_oof
        else:
            output_R = torch.squeeze(output_R.data.cpu(), 1)[:, :, :]
            mask_max_depth_R = output_R < 4.527
            mask_min_depth_R = output_R > 0.494
            mask_R = mask_max_depth_R & mask_min_depth_R
            mask_R = torch.ones_like(mask_R)

    # stereo_L = torch.squeeze(output_stereo.data.cpu(), 1)[:, :, :]
    stereo_L = torch.squeeze(output_stereo.data, 1)[:, :, :]
    # mono_L = torch.unsqueeze(mono_L, 0).cpu()
    mono_L = torch.unsqueeze(mono_L, 0)
    mono_R = torch.unsqueeze(mono_R, 0)
    mono_R = torch.unsqueeze(mono_R, 0)
    output_stereo = torch.unsqueeze(output_stereo, 0)
    mono_RtoL = apply_disparity(mono_R, -output_stereo)
    mono_RtoL = torch.squeeze(mono_RtoL,0)
    output_stereo = torch.squeeze(output_stereo, 0)

    output_fuse = output_stereo.clone()
    output_fuse[fuse_mask_700] = mono_L[fuse_mask_700]
    output_fuse[fuse_mask_1500 & torch.abs(left_occ - 1).byte().to(device)]  = mono_RtoL[fuse_mask_1500 & torch.abs(left_occ - 1).byte().to(device)]
    # output_fuse[fuse_mask_1500]  = mono_RtoL[fuse_mask_1500]

    # output_fuse = torch.squeeze(output_fuse.data.cpu(), 1)[:, :, :]
    # mono_L = torch.squeeze(mono_L.data.cpu(), 1)[:, :, :]
    output_fuse_occ = output_fuse.clone()
    output_fuse_oof = output_fuse.clone()
    output_fuse_occ_oof = output_fuse.clone()
    output_fuse_occ[occ_mask] = mono_L[occ_mask]
    output_fuse_oof[oof_mask] = mono_L[oof_mask]
    output_fuse_occ_oof[occ_oof_mask] = mono_L[occ_oof_mask]

    # plt.subplot(231)
    # plt.title("GT")
    # plt.imshow(disp_true[0], vmin=0, vmax=10, cmap='jet')
    # plt.subplot(232)
    # plt.title("Fuse Mask")
    # plt.imshow(fuse_mask_1500[0] | fuse_mask_700[0], vmin=0, vmax=1, cmap='jet')
    # plt.subplot(233)
    # plt.title("Stereo Output")
    # plt.imshow(output_stereo[0], vmin=0, vmax=10, cmap='jet')
    # plt.subplot(234)
    # plt.title("Mono Left Output")
    # plt.imshow(mono_L[0], vmin=0, vmax=10, cmap='jet')
    # # plt.subplot(235)
    # # plt.title("Mono RtoL output")
    # # plt.imshow(mono_RtoL[0], vmin=0, vmax=10, cmap='jet')
    # plt.subplot(236)
    # plt.title("Fuse output")
    # plt.imshow(output_fuse[0], vmin=0, vmax=10, cmap='jet')


    # mask_max_dis = output < 192
    # mask_min_dis = output > 4.85
    # mask_max_depth = stereo_L < 4.527
    # mask_min_depth = stereo_L > 0.494
    # mask = mask_max_depth & mask_min_depth

    # hist = list()


    if len(disp_true[mask]) == 0:
        stereo_loss = 0
    else:
        # mask_psi_range = mask_psi_max & mask_psi_min
        # Stereo Loss
        stereo_loss = torch.mean(torch.abs(stereo_L[mask] - disp_true[mask]))  # end-point-error
        rel_stereo_loss = torch.mean(torch.abs(stereo_L[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error
        rel_stereo_loss_masked = torch.mean(
            torch.abs(stereo_L[fuse_mask_700] - disp_true[fuse_mask_700]) / disp_true[fuse_mask_700])  # end-point-error
        # Mono Loss
        rel_mono_loss = torch.mean(torch.abs(mono_L[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error
        rel_mono_loss_masked = torch.mean(
            torch.abs(mono_L[fuse_mask_700] - disp_true[fuse_mask_700]) / disp_true[fuse_mask_700])  # end-point-error

        # Mono RtoL Loss
        rel_mono_RtoL_loss = torch.mean(torch.abs(mono_RtoL[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error
        rel_mono__RtoL_loss_masked = torch.mean(
            torch.abs(mono_RtoL[fuse_mask_1500] - disp_true[fuse_mask_1500]) / disp_true[fuse_mask_1500])  # end-point-error


        # Fused loss
        rel_loss_fused = torch.mean(torch.abs(output_fuse[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error

        loss_R = 0
        rel_loss_R = 0

        rel_loss_fused_on_oof = torch.mean(
            torch.abs(output_fuse_oof[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error
        rel_loss_fused_on_occ = torch.mean(
            torch.abs(output_fuse_occ[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error
        rel_loss_fused_on_occ_oof = torch.mean(
            torch.abs(output_fuse_occ_oof[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error

        # Rel error in occ and oof, in depth ranges inside psi
        rel_stereo_loss_occ = torch.mean(
            torch.abs(stereo_L[occ_mask] - disp_true[occ_mask]) / disp_true[occ_mask])  # end-point-error
        rel_mono_loss_occ = torch.mean(
            torch.abs(mono_L[occ_mask] - disp_true[occ_mask]) / disp_true[occ_mask])  # end-point-error
        rel_stereo_loss_oof = torch.mean(
            torch.abs(stereo_L[oof_mask] - disp_true[oof_mask]) / disp_true[oof_mask])  # end-point-error
        rel_mono_loss_oof = torch.mean(
            torch.abs(mono_L[oof_mask] - disp_true[oof_mask]) / disp_true[oof_mask])  # end-point-error

        if args.right_head and not args.pred_occlusion:
            loss_R = torch.mean(torch.abs(output_R[mask_R] - disp_true_R[mask_R]))  # end-point-error
            rel_loss_R = torch.mean(
                torch.abs(output_R[mask_R] - disp_true_R[mask_R]) / disp_true_R[mask_R])  # end-point-error
        # hist.append(((disp_true_R[mask_R]*100).long(), (torch.abs(output_R[mask_R] - disp_true_R[mask_R]) / disp_true_R[mask_R])))

        stereo_err = (torch.abs(stereo_L[mask] - disp_true[mask]) / disp_true[mask]).cpu().numpy()
        mono_err   = (torch.abs(mono_L[mask] - disp_true[mask]) / disp_true[mask]).cpu().numpy()
        mono_RtoL_err = (torch.abs(mono_RtoL[mask] - disp_true[mask]) / disp_true[mask]).cpu().numpy()

        stereo_err_for_occluded = (torch.abs(stereo_L[occ_mask] - disp_true[occ_mask]) / disp_true[occ_mask]).cpu().numpy()
        mono_err_for_occluded = (torch.abs(mono_L[occ_mask] - disp_true[occ_mask]) / disp_true[occ_mask]).cpu().numpy()
        stereo_err_for_oof = (torch.abs(stereo_L[oof_mask] - disp_true[oof_mask]) / disp_true[oof_mask]).cpu().numpy()
        mono_err_for_oof = (torch.abs(mono_L[oof_mask] - disp_true[oof_mask]) / disp_true[oof_mask]).cpu().numpy()

        depth_for_occ = (disp_true[occ_mask] * 100).long().cpu().numpy()
        depth_for_oof = (disp_true[oof_mask] * 100).long().cpu().numpy()
        depth_all     = (disp_true[mask] * 100).long().cpu().numpy()

        for i, dpt in tqdm.tqdm(enumerate(depth_all)):
            # dpt = (dpt * 100).long().item()
            stereo_err_for_depth[dpt] += stereo_err[i]
            mono_err_for_depth[dpt] += mono_err[i]
            mono_RtoL_err_for_depth[dpt] += mono_RtoL_err[i]
            all_depth_cnt[dpt] += 1

        for i, dpt in tqdm.tqdm(enumerate(depth_for_occ)):
            # dpt = (dpt * 100).long().item()
            stereo_occ_err_for_depth[dpt] += stereo_err_for_occluded[i]
            mono_occ_err_for_depth[dpt] += mono_err_for_occluded[i]
            occ_depth_cnt[dpt] += 1

        for i, dpt in enumerate(depth_for_oof):
            # dpt = (dpt * 100).long().item()
            stereo_oof_err_for_depth[dpt] += stereo_err_for_oof[i]
            mono_oof_err_for_depth[dpt] += mono_err_for_oof[i]
            oof_depth_cnt[dpt] += 1

    return stereo_loss, rel_stereo_loss, rel_stereo_loss_masked, rel_mono_loss, rel_mono_loss_masked, rel_loss_fused, loss_R, rel_loss_R, rel_loss_fused_on_oof, rel_loss_fused_on_occ,rel_loss_fused_on_occ_oof, rel_stereo_loss_occ, rel_mono_loss_occ, rel_stereo_loss_oof, rel_mono_loss_oof


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (1 / epoch)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def show_occ_oof_histogram(occ=True):
    depth_histo = dict()
    occ_depth_histo = dict()
    for batch_idx, (imgL, imgR, disp_L, disp_R, left_occ, left_oof) in tqdm.tqdm(enumerate(TestImgLoader)):
        for dpt in disp_L.view(-1):
            dpt = (dpt * 100).long().item()
            if dpt in depth_histo:
                depth_histo[dpt] += 1
            else:
                depth_histo[dpt] = 1
        if occ:
            masked_depth = disp_L[left_occ == 1].view(-1)
        else:
            masked_depth = disp_L[left_oof == 1].view(-1)
        for dpt in masked_depth:
            dpt = (dpt * 100).long().item()
            if dpt in occ_depth_histo:
                occ_depth_histo[dpt] += 1
            else:
                occ_depth_histo[dpt] = 1
    depths_keys, depth_values = zip(*sorted(zip(depth_histo.keys(), depth_histo.values())))
    occ_depths_keys, occ_depth_values = zip(*sorted(zip(occ_depth_histo.keys(), occ_depth_histo.values())))
    plt.plot(depths_keys, depth_values, label="All Depth")
    plt.plot(occ_depths_keys, occ_depth_values, label="Occlusion Depth")
    plt.legend()
    plt.show()


def main():
    start_full_time = time.time()

    # ------------- TEST ------------------------------------------------------------
    total_stereo_loss = 0
    total_stereo_rel_loss = 0
    total_stereo_rel_loss_masked = 0
    total_mono_rel_loss = 0
    total_mono_rel_loss_masked = 0
    total_fuse_rel_loss = 0
    rel_mono_losses_occ = 0
    rel_stereo_losses_occ = 0
    rel_mono_losses_oof = 0
    rel_stereo_losses_oof = 0
    total_fuse_on_occ_rel_loss = 0
    total_fuse_on_oof_rel_loss = 0
    total_stereo_loss_R = 0
    total_stereo_rel_loss_R = 0
    hist = list()
    stereo_masked_rel_loss_cnt = 0
    mono_masked_rel_loss_cnt = 0
    rel_occ_losses_cnt = 0
    rel_oof_losses_cnt = 0
    total_fuse_on_occ_oof_rel_loss = 0
    # show_occ_oof_histogram(occ=True)
    # show_occ_oof_histogram(occ=False)

    for batch_idx, (imgL, imgR, disp_L, disp_R, left_occ, left_oof) in enumerate(TestImgLoader):
        stereo_loss, rel_stereo_loss, rel_stereo_loss_masked, rel_mono_loss, \
        rel_mono_loss_masked, rel_loss_fused, loss_R, rel_loss_R, rel_loss_fused_on_oof, \
        rel_loss_fused_on_occ, rel_loss_fused_on_occ_oof, rel_stereo_loss_occ, rel_mono_loss_occ, rel_stereo_loss_oof, rel_mono_loss_oof = test(
            imgL, imgR, disp_L, disp_R, left_occ, left_oof)

        # Stereo Loss
        print('Iter %d Stereo test loss = %.3f' % (batch_idx, stereo_loss))
        print('Iter %d Stereo test rel loss = %.3f' % (batch_idx, rel_stereo_loss))
        print('Iter %d Stereo test rel loss masked = %.3f' % (batch_idx, rel_stereo_loss_masked))
        # Mono Loss
        print('Iter %d Mono test rel loss = %.3f' % (batch_idx, rel_mono_loss))
        print('Iter %d Mono test rel loss masked = %.3f' % (batch_idx, rel_mono_loss_masked))

        # Fuse Loss
        print('Iter %d Fused test rel loss = %.3f' % (batch_idx, rel_loss_fused))

        print('Iter %d Fused on occ test rel loss = %.3f' % (batch_idx, rel_loss_fused_on_occ))
        print('Iter %d Fused on oof test rel loss = %.3f' % (batch_idx, rel_loss_fused_on_oof))
        print('Iter %d Fused on occ and oof test rel loss = %.3f' % (batch_idx, rel_loss_fused_on_occ_oof))


        if args.right_head and not args.pred_occlusion:
            print('Iter %d Stereo test loss right image = %.3f' % (batch_idx, loss_R))
            print('Iter %d Stereo test rel loss right image = %.3f' % (batch_idx, rel_loss_R))

        print('Iter %d Stereo test rel loss for occluded pixels = %.3f' % (batch_idx, rel_stereo_loss_occ))
        print('Iter %d Mono test rel loss for occluded pixels = %.3f' % (batch_idx, rel_mono_loss_occ))
        print('Iter %d Stereo test rel loss for out-of-frame pixels = %.3f' % (batch_idx, rel_stereo_loss_oof))
        print('Iter %d Mono test rel mono loss for out-of-frame pixels = %.3f' % (batch_idx, rel_mono_loss_oof))

        # plt.show()

        total_stereo_loss += stereo_loss
        total_stereo_rel_loss += rel_stereo_loss
        if not torch.isnan(rel_stereo_loss_masked):
            total_stereo_rel_loss_masked += rel_stereo_loss_masked
            stereo_masked_rel_loss_cnt += 1
        total_mono_rel_loss += rel_mono_loss
        if not torch.isnan(rel_mono_loss_masked):
            total_mono_rel_loss_masked += rel_mono_loss_masked
            mono_masked_rel_loss_cnt += 1
        total_fuse_rel_loss += rel_loss_fused

        total_fuse_on_oof_rel_loss += rel_loss_fused_on_oof
        total_fuse_on_occ_rel_loss += rel_loss_fused_on_occ
        total_fuse_on_occ_oof_rel_loss += rel_loss_fused_on_occ_oof

        if args.right_head and not args.pred_occlusion:
            total_stereo_loss_R += loss_R
            total_stereo_rel_loss_R += rel_loss_R

        if not torch.isnan(rel_stereo_loss_occ):
            rel_stereo_losses_occ += rel_stereo_loss_occ
            rel_mono_losses_occ += rel_mono_loss_occ
            rel_occ_losses_cnt += 1
        if not torch.isnan(rel_stereo_loss_oof):
            rel_stereo_losses_oof += rel_stereo_loss_oof
            rel_mono_losses_oof += rel_mono_loss_oof
            rel_oof_losses_cnt += 1
        # hist.append(rel_loss_histo)

    # Stereo Loss        
    print('Total Stereo test loss = %.3f' % (total_stereo_loss / len(TestImgLoader)))
    print('Total Stereo test rel loss = %.3f' % (total_stereo_rel_loss / len(TestImgLoader)))
    print('Total Stereo test rel loss masked = %.3f' % (total_stereo_rel_loss_masked / stereo_masked_rel_loss_cnt))
    # Mono Loss
    print('Total Mono test rel loss = %.3f' % (total_mono_rel_loss / len(TestImgLoader)))
    print('Total Mono test rel loss masked = %.3f' % (total_mono_rel_loss_masked / mono_masked_rel_loss_cnt))

    # Fuse Loss
    print('Total Fused test rel loss = %.3f' % (total_fuse_rel_loss / len(TestImgLoader)))

    print('Total Fused on oof test rel loss = %.3f' % (total_fuse_on_oof_rel_loss / len(TestImgLoader)))
    print('Total Fused on occ test rel loss = %.3f' % (total_fuse_on_occ_rel_loss / len(TestImgLoader)))
    print('Total Fused on occ and oof test rel loss = %.3f' % (total_fuse_on_occ_oof_rel_loss / len(TestImgLoader)))

    if args.right_head and not args.pred_occlusion:
        print('Total Stereo test loss right image = %.3f' % (total_stereo_loss_R / len(TestImgLoader)))
        print('Total Stereo test rel loss right image = %.3f' % (total_stereo_rel_loss_R / len(TestImgLoader)))

    print('Total Stereo test rel loss for occluded pixels = %.3f' % (rel_stereo_losses_occ / rel_occ_losses_cnt))
    print('Total Mono test rel loss for occluded pixels = %.3f' % (rel_mono_losses_occ / rel_occ_losses_cnt))
    print('Total Stereo test rel loss for out-of-frame pixels = %.3f' % (rel_stereo_losses_oof / rel_oof_losses_cnt))
    print('Total Mono test rel mono loss for out-of-frame pixels = %.3f' % (rel_mono_losses_oof / rel_oof_losses_cnt))

    all_depth = [i[0] for i in all_depth_cnt.items() if i[1] != 0]
    occ_depth = [i[0] for i in occ_depth_cnt.items() if i[1] != 0]
    oof_depth = [i[0] for i in oof_depth_cnt.items() if i[1] != 0]

    all_depth_cnt_l = np.array([i[1] for i in all_depth_cnt.items() if i[1] != 0])
    occ_depth_cnt_l = np.array([i[1] for i in occ_depth_cnt.items() if i[1] != 0])
    oof_depth_cnt_l = np.array([i[1] for i in oof_depth_cnt.items() if i[1] != 0])

    stereo_err = np.array([stereo_err_for_depth[i] for i in all_depth])
    mono_err   = np.array([mono_err_for_depth[i] for i in all_depth])
    mono_RtoL_err   = np.array([mono_RtoL_err_for_depth[i] for i in all_depth])


    stereo_occ_err = np.array([stereo_occ_err_for_depth[i] for i in occ_depth])
    mono_occ_err = np.array([mono_occ_err_for_depth[i] for i in occ_depth])

    stereo_oof_err = np.array([stereo_oof_err_for_depth[i] for i in oof_depth])
    mono_oof_err = np.array([mono_oof_err_for_depth[i] for i in oof_depth])

    plt.plot(all_depth, ((all_depth_cnt_l).astype(np.float) / np.sum(all_depth_cnt_l))*100, label="Depth Count Percentage")
    plt.plot(all_depth, stereo_err / all_depth_cnt_l, label="Stereo Rel Error")
    plt.plot(all_depth, mono_err / all_depth_cnt_l, label="Mono Rel Error")
    plt.plot(all_depth, mono_RtoL_err / all_depth_cnt_l, label="Mono RtoL Rel Error")
    plt.plot(occ_depth, stereo_occ_err / occ_depth_cnt_l, label='Stereo Occ Rel Error')
    plt.plot(occ_depth, mono_occ_err / occ_depth_cnt_l, label='Mono Occ Rel Error')
    plt.plot(oof_depth, stereo_oof_err / oof_depth_cnt_l, label='Stereo Oof Rel Error')
    plt.plot(oof_depth, mono_oof_err / oof_depth_cnt_l, label='Mono Oof Rel Error')
    plt.legend()
    plt.show()
    # with open('error_stereo.pickle', 'wb') as f:
    #     pickle.dump(hist, f)
    # ----------------------------------------------------------------------------------
    # SAVE test information
    # savefilename = args.savemodel + 'testinformation.tar'
    # torch.save({
    #     'stereo_loss': total_stereo_loss / len(TestImgLoader),
    # }, savefilename)


if __name__ == '__main__':
    main()

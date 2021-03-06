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
# from dataloader import sintel_listflowfile_with_filter_with_depth as lt
# from dataloader import sintel_listflowfile_without_filter_with_depth as lt
from dataloader import SintelFlowLoader as DA
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo',
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
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
# parser.add_argument('--loadmodel', default='./checkpoints/checkpoint_9.tar',
                    help='load model')
# parser.add_argument('--loadmodel', default='./checkpoints/checkpoint_filter_loss_2.6.tar',
# parser.add_argument('--loadmodel', default='./checkpoints/checkpoint_clean_from_scratch_loss_2.1.tar',
# parser.add_argument('--loadmodel', default='./pretrained_model_KITTI2015.tar',
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
                    help='add right disparity head')
parser.add_argument('--pred_occlusion', action='store_true', default=False,
                    help='pred occlusion or right depth')
parser.add_argument('--pred_oof', action='store_true', default=False,
                    help='pred out of frame head')

# parser.add_argument('--gpu_id', type=int, default=0, metavar='S',
#                     help='gpu_id')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, all_right_disp, all_left_occ, all_left_oof, test_left_img, test_right_img, test_left_disp, test_right_disp, test_left_occ, test_left_oof= lt.dataloader(args.datapath, args.left_imgs, args.right_imgs, args.disp_imgs,args.disp_R_imgs, occ_L_dir=args.occ_L_dir, oof_L_dir=args.oof_L_dir,clean=args.clean)

TrainImgLoader = torch.utils.data.DataLoader(
    # DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, dploader=DA.depth_loader),
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, all_right_disp, left_occ=all_left_occ, left_oof=all_left_oof, training=True, dploader=depth_read, cont=args.cont),
    batch_size=1, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, test_right_disp, left_occ=test_left_occ, left_oof=test_left_oof, training=False, dploader=depth_read, cont=args.cont),
    batch_size=1, shuffle=False, num_workers=4, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, device=device, dfd_net=args.dfd, dfd_at_end=args.dfd_at_end, right_head=args.right_head, pred_oof=args.pred_oof)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.to(device)

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
# optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L, disp_R,left_occ, left_oof):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true, disp_R, left_occ, left_oof = imgL.to(device), imgR.to(device), disp_L.to(device), disp_R.to(device), left_occ.to(device), left_oof.to(device)

    # ---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    mask_R = disp_R < args.maxdisp
    mask_R.detach_()
    # ----
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        if args.pred_oof:
            output1, output2, output3, output4, output5, output6,output7, output8, output9 = model(imgL, imgR)
        if args.right_head:
            output1, output2, output3, output4, output5, output6= model(imgL, imgR)
        else:
            output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        if args.right_head:
            output4 = torch.squeeze(output4, 1)
            output5 = torch.squeeze(output5, 1)
            output6 = torch.squeeze(output6, 1)
        if args.pred_oof:
            output7 = torch.squeeze(output7, 1)
            output8 = torch.squeeze(output8, 1)
            output9 = torch.squeeze(output9, 1)
        # loss = F.smooth_l1_loss(output3[mask], disp_true[mask],size_average=True)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],size_average=True)
        if args.right_head:
            if args.pred_occlusion:
                loss += 0.5 * F.smooth_l1_loss(output4[mask], left_occ[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
                    output5[mask], left_occ[mask], size_average=True) + F.smooth_l1_loss(output6[mask], left_occ[mask],  size_average=True)
                if args.pred_oof:
                    loss += 0.5 * F.smooth_l1_loss(output7[mask], left_oof[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
                        output8[mask], left_oof[mask], size_average=True) + F.smooth_l1_loss(output9[mask], left_oof[mask], size_average=True)
            else:
                loss += 0.5 * F.smooth_l1_loss(output4[mask_R], disp_R[mask_R], size_average=True) + 0.7 * F.smooth_l1_loss(
                    output5[mask_R], disp_R[mask_R], size_average=True) + F.smooth_l1_loss(output6[mask_R], disp_R[mask_R],size_average=True)


    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.to(device), imgR.to(device)

    # ---------
    mask = disp_true < 192
    # ----

    with torch.no_grad():
        output3 = model(imgL, imgR)

    output = torch.squeeze(output3.data.cpu(), 1)[:, 4:, :]

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error

    return loss


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001*(1/epoch)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0
        # adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, disp_crop_R,left_occ, left_oof) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L, disp_crop_R, left_occ, left_oof)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))

        # SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    # ------------- TEST ------------------------------------------------------------
    # total_test_loss = 0
    # for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
    #     test_loss = test(imgL, imgR, disp_L)
    #     print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
    #     total_test_loss += test_loss
    #
    # print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
    # ----------------------------------------------------------------------------------
    # SAVE test information
    # savefilename = args.savemodel + 'testinformation.tar'
    # torch.save({
    #     'test_loss': total_test_loss / len(TestImgLoader),
    # }, savefilename)


if __name__ == '__main__':
    main()

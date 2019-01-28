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
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# test_imgs  = ['City_R_0116_1100_maskImg.png', 'City_R_0092_1100_maskImg.png', 'City_R_0114_1100_maskImg.png', 'City_R_0190_1100_maskImg.png', 'City_R_0154_1100_maskImg.png', 'City_R_0066_1100_maskImg.png', 'City_R_0202_1100_maskImg.png', 'City_R_0204_1100_maskImg.png', 'City_R_0128_1100_maskImg.png', 'City_R_0042_1100_maskImg.png', 'City_R_0026_1100_maskImg.png', 'City_R_0058_1100_maskImg.png']

# if args.clean:
#     test_imgs = [img.replace('_1100_maskImg.png', '.tif') for img in test_imgs]

# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath, args.left_imgs, args.right_imgs, args.disp_imgs,filenames=test_imgs, clean=args.clean)
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath, args.left_imgs, args.right_imgs, args.disp_imgs, clean=args.clean)



TrainImgLoader = torch.utils.data.DataLoader(
    # DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True,dploader=DA.depth_loader),
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, dploader=depth_read,cont=args.cont),
    batch_size=1, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    # DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, dploader=DA.depth_loader),
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, dploader=depth_read, cont=args.cont),
    batch_size=2, shuffle=False, num_workers=4, drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, device=device, dfd_net=args.dfd, dfd_at_end=args.dfd_at_end)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    # ----
    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                  size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data[0]

def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))

    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    # ---------
    mask = disp_true < 192
    # ----

    with torch.no_grad():
        output3 = model(imgL, imgR)

    output = torch.squeeze(output3.data.cpu(), 1)[:, :, :]
    disp_true = disp_true[:,36:-36,:]
    # mask_max_dis = output < 192
    # mask_min_dis = output > 4.85
    mask_max_depth = output < 4.527
    mask_min_depth = output > 0.494
    mask = mask_max_depth & mask_min_depth

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error
        rel_loss = torch.mean(torch.abs(output[mask] - disp_true[mask]) / disp_true[mask])  # end-point-error
    return loss, rel_loss


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001*(1/epoch)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()

    # ------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    total_test_rel_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss, rel_loss = test(imgL, imgR, disp_L)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
        print('Iter %d test rel loss = %.3f' % (batch_idx, rel_loss))
        total_test_loss += test_loss
        total_test_rel_loss += rel_loss

    print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
    print('total test rel loss = %.3f' % (total_test_rel_loss / len(TestImgLoader)))
    # ----------------------------------------------------------------------------------
    # SAVE test information
    # savefilename = args.savemodel + 'testinformation.tar'
    # torch.save({
    #     'test_loss': total_test_loss / len(TestImgLoader),
    # }, savefilename)


if __name__ == '__main__':
    main()

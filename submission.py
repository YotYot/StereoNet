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
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *
import matplotlib.pyplot as plt
from SintelFlowLoader import default_loader
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Stereo/',
                    help='select model')
parser.add_argument('--left_dir', default=None,
                    help='select left dir')
parser.add_argument('--right_dir', default=None,
                    help='select right dir')
parser.add_argument('--loadmodel', default=None,
# parser.add_argument('--loadmodel', default='./checkpoints/checkpoint_clean_from_scratch_loss_2.1.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--outdir', default='default',
                    help='output dir')
parser.add_argument('--dfd', action='store_true', default=False,
                    help='include dfd net')
parser.add_argument('--dfd_at_end', action='store_true', default=False,
                    help='include dfd net')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA

import sintel_loader as DA

test_left_img, test_right_img = DA.dataloader(args.datapath, args.left_dir, args.right_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, device=device, dfd_net=args.dfd, dfd_at_end=args.dfd_at_end)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp


def main():
   processed = preprocess.get_transform(augment=False)

   for inx in range(len(test_left_img)):

       # imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
       # imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
       imgL_o = np.array(default_loader(test_left_img[inx]))
       imgR_o = np.array(default_loader(test_right_img[inx]))
       imgL = processed(imgL_o).numpy()
       imgR = processed(imgR_o).numpy()
       imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
       imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

       # pad to (384, 1248)
       top_pad = 512-imgL.shape[2]
       # top_pad = 384-imgL.shape[2]
       # left_pad = 1248-imgL.shape[3]
       # imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       # imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,0)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,0)),mode='constant',constant_values=0)

       start_time = time.time()
       with torch.no_grad():
            pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       top_pad   = 512-imgL_o.shape[0]
       # top_pad   = 384-imgL_o.shape[0]
       # left_pad  = 1248-imgL_o.shape[1]
       disparity_dir = '/home/yotamg/data/sintel_depth/training/disparities_viz/'
       # file_splits = test_left_img[inx].split('/')[-1].split("_frame_")
       # a = plt.imread(os.path.join(disparity_dir, file_splits[0],'frame_' + file_splits[1]))
       img = pred_disp[top_pad:,:]
       img = 1 / img
       # plt.figure(1)
       # plt.subplot(1,2,1)
       # plt.imshow(img)
       # plt.subplot(1,2,2)
       # plt.imshow(a)
       # plt.show()
       outdir = os.path.join('./outputs', args. outdir)
       if not os.path.isdir(outdir):
           os.makedirs(outdir)
       plt.imsave(os.path.join(outdir, test_left_img[inx].split('/')[-1]), (img*256).astype('uint16'), cmap='jet')
       # plt.imsave(os.path.join(outdir, test_left_img[inx].split('/')[-1]), (img*256).astype('uint16'),cmap='gray')
       # skimage.io.imsave(test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))

if __name__ == '__main__':
   main()







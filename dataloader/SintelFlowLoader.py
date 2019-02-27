import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import preprocess 
import listflowfile as lt
import readpfm as rp
import numpy as np
from sintel_io import disparity_read
from sintel_io import depth_read
from local_utils import noisy


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def occ_oof_loader(path):
    return Image.open(path).convert('L')

def disparity_loader(path):
    # return rp.readPFM(path)
    return disparity_read(path)

def depth_loader(path):
    # return rp.readPFM(path)
    return depth_read(path)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, right_disparity, left_occ=None, left_oof=None, training=True, loader=default_loader, dploader= disparity_loader, occ_loader=occ_oof_loader, cont=False, submission=False):
 
        self.left = left
        self.right = right
        if not submission:
            self.disp_L = left_disparity
            self.disp_R = right_disparity
        self.left_occ = left_occ
        if left_occ:
            self.occ_L = left_occ
            self.oof_L = left_oof
        self.loader = loader
        self.dploader = dploader
        self.occ_loader = occ_loader
        self.training = training
        self.cont = cont
        self.submission = submission


    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        if not self.submission:
            disp_L= self.disp_L[index]
            disp_R = self.disp_R[index]


        left_img = self.loader(left)
        right_img = self.loader(right)

        if not self.submission:
            dataL = self.dploader(disp_L)
            dataL = np.ascontiguousarray(dataL,dtype=np.float32)
            dataR = self.dploader(disp_R)
            dataR = np.ascontiguousarray(dataR, dtype=np.float32)
            if self.left_occ:
                left_occ = self.occ_loader(self.occ_L[index])
                left_oof = self.occ_loader(self.oof_L[index])
                left_occ = torch.squeeze(transforms.ToTensor()(left_occ), 0)
                left_oof = torch.squeeze(transforms.ToTensor()(left_oof), 0)
            #Fix psi from -4:10 to 1:15
            # if self.cont:
                # dataL = 23.46 / dataL
                # dataL += 5
                # #Fixing to be more similar to disparity
                # dataL *= 10

        if self.training:
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = dataL[y1:y1 + th, x1:x1 + tw]
           dataR = dataR[y1:y1 + th, x1:x1 + tw]

           if self.left_occ:
            left_occ = left_occ[y1:y1 + th, x1:x1 + tw]
            left_oof = left_oof[y1:y1 + th, x1:x1 + tw]


           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           if self.left_occ:
               return left_img, right_img, dataL, dataR, left_occ, left_oof
           else:
               return left_img, right_img, dataL, dataR
        else:
           w, h = left_img.size
           left_img = left_img.crop((w-1024, h-512, w, h))
           right_img = right_img.crop((w-1024, h-512, w, h))

           processed = preprocess.get_transform(augment=False)
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           # left_img  += torch.Tensor(noisy(left_img))
           # right_img += torch.Tensor(noisy(right_img))

           left_img = torch.clamp(left_img, -1, 1)
           right_img = torch.clamp(right_img, -1, 1)


           if self.submission:
               return left_img,right_img
           elif self.left_occ:
               return left_img, right_img, dataL, dataR, left_occ, left_oof
           else:
               return left_img, right_img, dataL, dataR

    def __len__(self):
        return len(self.left)

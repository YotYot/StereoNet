import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, left_dir,right_dir):

  left_fold  = left_dir + '/'
  right_fold = right_dir + '/'

  # test_fold = 'clean_left_flatten_testing/'

  image = [img for img in os.listdir(filepath+right_dir) if is_image_file(img)]


  # left_test  = [filepath+left_fold+img.replace('_R_','_').replace('.png','_1100_maskImg.png') for img in image]
  left_test  = [filepath+left_fold+img.replace('_R_','_') for img in image]
  right_test = [filepath+right_fold+img for img in image]

  return left_test, right_test

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

 # left_train_dir = os.path.join(filepath,'training', 'clean_left_flatten/')
 left_train_dir = os.path.join(filepath, 'clean_left_flatten_filter_training/')
 right_train_dir = os.path.join(filepath, 'clean_right_flatten/')
 disp_train_dir = os.path.join(filepath, 'disparities_flatten/')
 left_test_dir = os.path.join(filepath, 'clean_left_flatten_filter_testing/')
 right_test_dir = os.path.join(filepath,  'clean_right_flatten/')
 disp_test_dir = os.path.join(filepath,  'disparities_flatten/')


 # all_left_img = os.listdir(left_train_dir)
 file_list = os.listdir(left_train_dir)
 all_left_img  = [left_train_dir  + s for s in file_list]
 all_right_img = [right_train_dir + s for s in file_list]
 all_left_disp = [disp_train_dir  + s for s in file_list]

 file_list = os.listdir(left_test_dir)
 test_left_img  = [left_test_dir  + s for s in file_list]
 test_right_img = [right_test_dir + s for s in file_list]
 test_left_disp = [disp_test_dir  + s for s in file_list]


 return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp



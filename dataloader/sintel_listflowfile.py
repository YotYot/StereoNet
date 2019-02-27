import torch.utils.data as data

from PIL import Image
import os
import os.path
import random
from operator import itemgetter

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(root, left_img_dir, right_img_dir, disp_dir, disp_R_dir, occ_L_dir=None, oof_L_dir=None, filenames=None, clean=False, test_only=False, focus_L='1500', focus_R='700'):

 left_img_dir = os.path.join(root,left_img_dir)
 right_img_dir = os.path.join(root,right_img_dir)
 test_right_img_dir = os.path.join(right_img_dir,'val')
 disp_dir = os.path.join(root,disp_dir)
 disp_R_dir = os.path.join(root, disp_R_dir)
 if occ_L_dir:
    occ_L_dir = os.path.join(root, occ_L_dir)
    oof_L_dir = os.path.join(root, oof_L_dir)

 right_img_filelist  = [os.path.join(right_img_dir,img)  for img in os.listdir(right_img_dir)]

 if filenames:
    test_right_img_filelist =  [os.path.join(right_img_dir,img)  for img in filenames]
 else:
    # test_right_img_filelist  = itemgetter(*inds)(right_img_filelist)
    if clean:
        test_right_img_filelist = [os.path.join(test_right_img_dir, img) for img in os.listdir(test_right_img_dir) if img.endswith('.tif')]
    else:
        test_right_img_filelist  = [os.path.join(test_right_img_dir ,img)  for img in os.listdir(test_right_img_dir) if img.endswith('.png')]
 # train_right_img_filelist = [img for img in right_img_filelist if img not in test_right_img_filelist]
 train_right_img_filelist = [img for img in right_img_filelist if img.endswith('.png') or img.endswith('.tif')]

 train_left_img_filelist= [img.replace(right_img_dir, left_img_dir).replace('_R_','_').replace(focus_R, focus_L) for img in train_right_img_filelist]
 test_left_img_filelist = [img.replace(test_right_img_dir, left_img_dir).replace('_R_','_').replace(focus_R, focus_L) for img in test_right_img_filelist]

 if clean:
     train_disp_filelist = [img.replace(left_img_dir, disp_dir).replace('.tif', '.dpt') for img in
                            train_left_img_filelist]
     test_disp_filelist = [img.replace(left_img_dir, disp_dir).replace('.tif', '.dpt') for img in
                           test_left_img_filelist]
     train_disp_R_filelist = [img.replace(right_img_dir, disp_R_dir).replace('val','').replace('.tif', '.dpt') for img in
                            train_right_img_filelist]
     test_disp_R_filelist = [img.replace(right_img_dir, disp_R_dir).replace('val','').replace('.tif', '.dpt') for img in
                           test_right_img_filelist]
 else:
     train_disp_filelist = [img.replace(left_img_dir, disp_dir).replace('_' + focus_L + '_maskImg.png', '.dpt') for img in
                            train_left_img_filelist]
     test_disp_filelist = [img.replace(left_img_dir, disp_dir).replace('_' + focus_L + '_maskImg.png', '.dpt') for img in
                           test_left_img_filelist]
     train_disp_R_filelist = [img.replace(right_img_dir, disp_R_dir).replace('_' + focus_R + '_maskImg.png', '.dpt') for img in
                              train_right_img_filelist]
     test_disp_R_filelist = [img.replace(right_img_dir, disp_R_dir).replace('/val','').replace('_' + focus_R + '_maskImg.png', '.dpt') for img in
                             test_right_img_filelist]
     if occ_L_dir:
         train_occ_filelist = [img.replace(right_img_dir, occ_L_dir).replace('/val', '').replace('_' + focus_R + '_maskImg.png', '.png').replace('_R', '') for img in
                               train_right_img_filelist]
         test_occ_filelist = [img.replace(right_img_dir, occ_L_dir).replace('/val','').replace('_' + focus_R + '_maskImg.png', '.png').replace('_R', '') for img in
                              test_right_img_filelist]
         train_oof_filelist = [img.replace(right_img_dir, oof_L_dir).replace('/val', '').replace('_' + focus_R + '_maskImg.png', '.png').replace('_R', '') for img in
                               train_right_img_filelist]
         test_oof_filelist = [img.replace(right_img_dir, oof_L_dir).replace('/val', '').replace('_' + focus_R + '_maskImg.png', '.png').replace('_R', '') for img in
                              test_right_img_filelist]


 if test_only:
     return test_left_img_filelist, test_right_img_filelist
 elif occ_L_dir:
     return train_left_img_filelist, train_right_img_filelist, train_disp_filelist, train_disp_R_filelist, train_occ_filelist, train_oof_filelist, test_left_img_filelist, test_right_img_filelist, test_disp_filelist, test_disp_R_filelist, test_occ_filelist, test_oof_filelist
 else:
    return train_left_img_filelist, train_right_img_filelist, train_disp_filelist, train_disp_R_filelist,test_left_img_filelist, test_right_img_filelist, test_disp_filelist, test_disp_R_filelist



import numpy as np
from sintel_io import disparity_read
import matplotlib.pyplot as plt
import os

def disp_a_to_disp_b(disp_a):
    a = disparity_read(disp_a)
    b = np.zeros_like(a)
    for i in range(436):
        for j in range(1024):
            x_loc = int(j - np.round(a[i,j]))
            if x_loc < 1024:
                b[i,x_loc] = a[i,j]
    plt.subplot(2,1,1)
    plt.imshow(a)
    plt.subplot(2,1,2)
    plt.imshow(b)

def depth_to_disp(src_dir, dst_dir):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    for depth in os.listdir(src_dir):
        depth_fullpath = os.path.join(src_dir,depth)
        dis_fullpath = os.path.join(dst_dir, depth)
        depth_img = plt.imread(depth_fullpath)
        dis = 1/depth_img
        plt.imsave(dis_fullpath,dis)


# disp_a_to_disp_b('/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/disparities_flatten/alley_1_frame_0001.png')
depth_to_disp('/home/yotamg/PycharmProjects/PSMNet/outputs/full_sintel_with_filter_depth', '/home/yotamg/PycharmProjects/PSMNet/outputs/full_sintel_with_filter_disp_from_depth')
print ("Done!")
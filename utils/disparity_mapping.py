#
# Author : Alwyn Mathew
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
# Bilinear sampler in pytorch(https://github.com/alwynmathew/bilinear-sampler-pytorch)
#

from __future__ import absolute_import, division, print_function
import torch
from torch.nn.functional import pad
import matplotlib.pyplot as plt
from sintel_io import disparity_read
import numpy as np
import time
import os
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = 'cpu'
def apply_disparity(input_images, x_offset, wrap_mode='border', tensor_type = 'torch.cuda.FloatTensor'):
    num_batch, num_channels, height, width = input_images.size()

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        input_images = pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == 'edge':
        edge_size = 0
    else:
        return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.view(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type).to(device)
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type).to(device)
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.view(-1).repeat(1, num_batch)
    y = y.view(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    # x = x + x_offset.contiguous().view(-1) * width
    x = x + x_offset.contiguous().view(-1)
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type).to(device)
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.view(num_channels, num_batch, height, width).permute(1,0,2,3)

    return output



def apply_disparity_modified(input_images, x_offset, wrap_mode='border', tensor_type='torch.cuda.FloatTensor'):
    num_batch, num_channels, height, width = input_images.size()

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        input_images = pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == 'edge':
        edge_size = 0
    else:
        return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.view(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type).to(device)
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type).to(device)
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.view(-1).repeat(1, num_batch)
    y = y.view(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    # x = x + x_offset.contiguous().view(-1) * width
    x = x + x_offset.contiguous().view(-1)
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type).to(device)
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # Occlusions mask
    mask = torch.zeros_like(x1)
    mul = torch.linspace(0, height - 1, height).repeat(width, 1).type(tensor_type).to(device).permute(1, 0)
    x_mul = x + width * mul.contiguous().view(1, -1)
    x_mul = torch.clamp(torch.round(x_mul).long(), 0, width * height - 1)
    mask[0][x_mul[0]] = 1

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = (weight_l * pix_l + weight_r * pix_r) * mask

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.view(num_channels, num_batch, height, width).permute(1, 0, 2, 3)

    return output


def apply_forward_disparity(input_images, x_offset, device, tensor_type='torch.cuda.FloatTensor'):
    cpu_device = 'cpu'
    x_offset = x_offset.to(cpu_device)
    input_images = input_images.to(cpu_device)
    time_before = time.time()
    num_batch, num_channels, height, width = input_images.size()
    left_image = torch.zeros_like(input_images)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type).to(cpu_device)
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type).to(cpu_device)
    offset_in_right = x - x_offset
    offset_in_right = torch.round(torch.clamp(offset_in_right, 0, width - 1)).long()
    left_image[:, :, y.long(), offset_in_right] = input_images
    left_image.to(device)
    print("Time: ", time.time() - time_before)
    return left_image


def generate_right(left_img, disp):
    right_img = torch.zeros_like(left_img)
    for i in range(right_img.shape[1]):
        for j in range(right_img.shape[2]):
            disp_x = torch.clamp(torch.floor(disp[i, j]).long(), 0, 1e9)
            right_img[:, i, j - disp_x] = left_img[:, i, j]

    return right_img


def generate_left(right_img, disp):
    left_img = torch.zeros_like(right_img).type('torch.cuda.FloatTensor').to(device)
    r_disp = left_disp_to_right_disp(disp)
    for i in range(left_img.shape[2]):
        for j in range(left_img.shape[3]):
            disp_x = torch.clamp(torch.floor(r_disp[i, j]).long(), 0, 1e9)
            left_img[:, :, i, j + disp_x] = right_img[:, :, i, j]
    return left_img


def left_disp_to_right_disp(left_disp):
    right_disp = torch.zeros_like(left_disp)
    for i in range(left_disp.shape[0]):
        for j in range(left_disp.shape[1]):
            right_offset = torch.clamp(j - (left_disp[i, j]).long(), 0, 1023)
            right_disp[i, right_offset] = left_disp[i, j]
    return right_disp


def example():
    dis = disparity_read(
        '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/disparities/alley_1/frame_0001.png')
    left_image = plt.imread(
        '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_left/alley_1/frame_0001.png')
    right_image = plt.imread(
        '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/clean_right/alley_1/frame_0001.png')
    occ = plt.imread(
        '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/occlusions/alley_1/frame_0001.png')
    oof = plt.imread(
        '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/outofframe/alley_1/frame_0001.png')
    ax1 = plt.subplot(221)
    plt.imshow(left_image)
    plt.subplot(222, sharex=ax1, sharey=ax1)
    plt.imshow(right_image)
    plt.subplot(223, sharex=ax1, sharey=ax1)
    plt.imshow(dis)
    dis = torch.Tensor(dis).to(device)
    # r_dis = left_disp_to_right_disp(dis)
    # plt.subplot(224, sharex=ax1, sharey=ax1)
    # plt.imshow(r_dis)
    left_image = torch.Tensor(np.transpose(left_image, (2, 0, 1)))
    left_image = torch.unsqueeze(left_image, 0).to(device)
    right_image = torch.Tensor(np.transpose(right_image, (2, 0, 1)))
    right_image = torch.unsqueeze(right_image, 0).to(device)
    # # b = generate_right(left_image,dis)
    # b = generate_left(right_image,dis)
    b = apply_disparity(right_image, -dis)
    occ = torch.abs(torch.Tensor(occ-1)).to(device)
    oof = torch.abs(torch.Tensor(oof-1)).to(device)
    b = b * occ * oof
        # b = apply_forward_disparity(right_image, -dis, device)
    plt.subplot(224, sharex=ax1, sharey=ax1)
    plt.imshow(b[0].permute(1, 2, 0))
    plt.show()
    print("Done")


if __name__ == '__main__':
    example()

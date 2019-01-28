#!/bin/bash
source ./venv2/bin/activate
export PYTHONPATH=/home/yotamg/PycharmProjects/PSMNet/dataloader:$PYTHONPATH
python test.py --left_imgs Tau_left_images/left_images_clean --right_imgs Tau_right_images/right_images_clean --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_clean_dfd/checkpoint_200.tar --clean --cont --dfd --savemodel clean_dfd
python test.py --left_imgs Tau_left_images/left_images_clean --right_imgs Tau_right_images/right_images_clean --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_clean_cont/checkpoint_100.tar --clean --cont --savemodel clean
python test.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_filtered_cont/checkpoint_100.tar --cont --savemodel filtered
python test.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_filtered_dfd/checkpoint_200.tar --cont --dfd --savemodel filtered_dfd
python test.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_filtered_dfd/checkpoint_200_dfd_no_init.tar --cont --dfd --savemodel filtered_dfd_no_init







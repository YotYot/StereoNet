#!/bin/bash

source ./venv/bin/activate
export PYTHONPATH=/home/yotamg/PycharmProjects/PSMNet/dataloader:$PYTHONPATH

python main.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_clean_cont    --clean --cont --epochs 100
python main.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_filtered_cont --cont --epochs 100

python test.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_clean_cont/checkpoint_100.tar --clean
python test.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_filtered_cont/checkpoint_100.tar

python main.py --left_imgs Tau_left_images/left_images_no_phase --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_no_phase_cont --cont --epochs 100
python test.py --left_imgs Tau_left_images/left_images_no_phase --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_no_phase_cont/checkpoint_100.tar







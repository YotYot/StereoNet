#!/bin/bash

source ./venv2/bin/activate
export PYTHONPATH=/home/yotamg/PycharmProjects/PSMNet/dataloader:$PYTHONPATH


## CONT GT -  NOT CONVERGING ###
# Run with cont PSI, with original PSM, for clean, filtered, no_phase
#python main.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/left_images_cont_GT --savemodel checkpoints_clean_cont_psi_PSM_only    --cont --epoch 200 --clean
#python main.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/left_images_cont_GT --savemodel checkpoints_filtered_cont_psi_PSM_only --cont --epoch 200
#python main.py --left_imgs Tau_left_images/left_images_no_phase --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/left_images_cont_GT --savemodel checkpoints_no_phase_cont_psi_PSM_only --cont --epoch 200
#
#python test.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/left_images_cont_GT --loadmodel ./checkpoints_clean_cont_psi_PSM_only/checkpoint_200.tar    --cont --clean
#python test.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/left_images_cont_GT --loadmodel ./checkpoints_filtered_cont_psi_PSM_only/checkpoint_200.tar --cont
#python test.py --left_imgs Tau_left_images/left_images_no_phase    --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/left_images_cont_GT --loadmodel ./checkpoints_no_phase_cont_psi_PSM_only/checkpoint_200.tar --cont

# Run with original depth, with original PSM, for clean, filtered, no_phase
python main.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_clean_orig_depth_PSM_only    --cont --epoch 100 --clean
python main.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_filtered_orig_depth_PSM_only --cont --epoch 100
python main.py --left_imgs Tau_left_images/left_images_no_phase --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_no_phase_orig_depth_PSM_only --cont --epoch 100
#
python test.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_clean_orig_depth_PSM_only/checkpoint_100.tar    --cont --clean
python test.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_filtered_orig_depth_PSM_only/checkpoint_100.tar --cont
python test.py --left_imgs Tau_left_images/left_images_no_phase  --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_no_phase_orig_depth_PSM_only/checkpoint_100.tar --cont


## Run with cont PSI, with PSM and DFD net, for clean, filtered, no_phase
#python main.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_clean_orig_depth_dfd    --cont --dfd --epoch 100 --clean
#python main.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_filtered_orig_depth_dfd --cont --dfd --epoch 100
#python main.py --left_imgs Tau_left_images/left_images_no_phase --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/original_depth --savemodel checkpoints_no_phase_orig_depth_dfd --cont --dfd --epoch 100


#python test.py --left_imgs Tau_left_images/left_images_clean    --right_imgs Tau_right_images/right_images_clean    --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_clean_orig_depth_dfd/checkpoint_100.tar    --dfd --cont --clean
#python test.py --left_imgs Tau_left_images/left_images_filtered --right_imgs Tau_right_images/right_images_filtered --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_filtered_orig_depth_dfd/checkpoint_100.tar --dfd --cont
#python test.py --left_imgs Tau_left_images/left_images_no_phase --right_imgs Tau_right_images/right_images_no_phase --disp_imgs Tau_left_images/original_depth --loadmodel ./checkpoints_no_phase_orig_depth_dfd/checkpoint_50.tar --dfd --cont








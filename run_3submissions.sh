#!/bin/bash

source ./venv/bin/activate
export PYTHONPATH=/home/yotamg/PycharmProjects/PSMNet/dataloader:$PYTHONPATH

python submission.py \
               --left_dir Tau_left_images/left_images_clean --right_dir Tau_right_images/right_images_clean --loadmodel ./checkpoints_clean/checkpoint_200.tar --outdir clean_result

python submission.py \
               --left_dir Tau_left_images/left_images_no_phase --right_dir Tau_right_images/right_images_no_phase --loadmodel ./checkpoints_no_phase/checkpoint_200.tar --outdir no_phase_result
python submission.py  \
               --left_dir Tau_left_images/left_images_filtered --right_dir Tau_right_images/right_images_filtered --loadmodel ./checkpoints_filtered//checkpoint_200.tar --outdir filtered_result






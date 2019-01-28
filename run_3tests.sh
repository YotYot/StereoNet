#!/bin/bash

python test.py --left_img_train_suffix clean_left_all_but_alley_1 \
               --left_img_test_suffix  clean_left_alley_1 \
               --savemodel ./trained/clean_left \
               --loadmodel ./trained/clean_left/checkpoint_50.tar

python test.py --left_img_train_suffix clean_left_filtered_all_but_alley_1 \
               --left_img_test_suffix  clean_left_filtered_alley_1 \
               --savemodel ./trained/filtered_left \
               --loadmodel ./trained/filtered_left/checkpoint_50.tar

python test.py --left_img_train_suffix clean_left_no_phase_all_but_alley_1 \
               --left_img_test_suffix  clean_left_no_phase_alley_1 \
               --savemodel ./trained/no_phase_left \
               --loadmodel ./trained/no_phase_left/checkpoint_50.tar







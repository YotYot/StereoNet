#!/bin/bash

python main.py --left_img_train_suffix clean_left_all_but_alley_1 \
               --left_img_test_suffix  clean_left_alley_1 \
               --savemodel ./trained/clean_left

python main.py --left_img_train_suffix clean_left_filtered_all_but_alley_1 \
               --left_img_test_suffix  clean_left_filtered_alley_1 \
               --savemodel ./trained/filtered_left

python main.py --left_img_train_suffix clean_left_no_phase_all_but_alley_1 \
               --left_img_test_suffix  clean_left_no_phase_alley_1 \
               --savemodel ./trained/no_phase_left

python test.py --loadmodel ./trained/clean_left --savemodel ./trained/clean_left
python test.py --loadmodel ./trained/filtered_left --savemodel ./trained/filtered_left
python test.py --loadmodel ./trained/no_phase_left --savemodel ./trained/no_phase_left







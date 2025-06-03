#!/bin/bash
python generator.py --train_images groundtruth_train --train_num_crops 10000 --destination_dir dataset_lores --max_workers 8 --resolution lores --palette 0 64 128 256 --rgb 444 666 --dither checkerboard atkinson None --crop_size 376 288 --rotate 10 20 30 40 45 50 60 70 80 --downscale 90 80 70 60 50 40 --verbose 1

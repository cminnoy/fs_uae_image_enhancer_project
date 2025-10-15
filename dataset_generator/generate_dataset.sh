#!/bin/bash
python generator.py \
    --train_images dataset/groundtruth_train \
    --train_num_crops 10 \
    --destination_dir dataset/dataset_train \
    --max_workers 4 \
    --resolution lores \
    --palette 0 32 64 128 256 512 1024 \
    --dither checkerboard floyd-steinberg atkinson sierra2 stucki burkes sierra3 bayer2x2 bayer4x4 bayer8x8 None \
    --crop_size 376 288 \
    --rotate 0 20 30 40 45 50 60 70 80 \
    --downscale 90 80 70 60 50 40 \
    --verbose 1

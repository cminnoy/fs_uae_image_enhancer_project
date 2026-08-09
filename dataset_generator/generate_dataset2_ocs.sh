#!/bin/bash
python generator.py \
    --train_images dataset2/groundtruth_train \
    --train_num_crops 6500 \
    --destination_dir dataset2/dataset_ocs \
    --max_workers 30 \
    --rgb 444 \
    --crop_size 752 576 \
    --resolution lores lores_laced hires hires_laced \
    --palette_algorithm median_cut \
    --palette 32 64 128 \
    --dither checkerboard floyd-steinberg atkinson sierra2 stucki burkes sierra3 bayer2x2 bayer4x4 bayer8x8 None \
    --verbose 1

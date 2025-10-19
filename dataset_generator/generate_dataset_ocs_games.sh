#!/bin/bash
python generator.py \
    --train_images dataset/groundtruth_train \
    --train_num_crops 15 \
    --destination_dir dataset/dataset_train_ocs_games \
    --max_workers 4 \
    --rgb 444 \
    --crop_size 376 288 \
    --rotate 0 1 2 3 4 5 6 7 8 9 10 \
             11 12 13 14 15 16 17 18 19 20 \
             21 22 23 24 25 26 27 28 29 30 \
             31 32 33 34 35 36 37 38 39 40 \
             41 42 43 44 45 46 47 48 49 50 \
             51 52 53 54 55 56 57 58 59 60 \
             61 62 63 64 65 66 67 68 69 70 \
             71 72 73 74 75 76 77 78 79 80 \
             81 82 83 84 85 86 87 88 89 90 \
    --downscale 90 80 70 60 50 40 \
    --resolution lores \
    --palette_algorithm median_cut \
    --palette 16 24 32 \
    --extra_mode EHB \
    --dither checkerboard floyd-steinberg atkinson sierra2 stucki burkes sierra3 bayer2x2 bayer4x4 bayer8x8 None \
    --verbose 1

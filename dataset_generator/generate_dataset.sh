#!/bin/bash
python generator.py --train_images groundtruth_train --train_num_crops 15000 --destination_dir dataset_edge_enhancer --max_workers 28 --resolution lores hires hires_laced lores_laced --crop_size 376 288 --keep_invalid_files --rotate 10 20 30 40 45 50 60 70 80 --downscale 90 80 70 60 50 --verbose 0

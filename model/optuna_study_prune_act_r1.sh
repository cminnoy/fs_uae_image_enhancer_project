#!/bin/bash
python optuna_study_prune_act_r1.py \
        --epochs 10 \
        --n_trials 35 \
        --study_name prune_act_r1 \
        --generator_train_dir ../dataset_generator/dataset_lores/ \
        --train_samples 10000 \
        --val_samples 1000 \
        --val_split_ratio 0.1 \
        --crop_size 376 288

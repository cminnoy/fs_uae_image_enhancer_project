#!/bin/bash
python optuna_study_opt_act.py \
        --epochs 10 \
        --n_trials 2 \
        --study_name opt_act_r1 \
        --generator_train_dir ../dataset_generator/dataset_lores/ \
        --train_samples 10000 \
        --val_samples 1000 \
        --val_split_ratio 0.1 \
        --crop_size 376 288 \
	--pruning_warmup_steps 6

#!/bin/bash
echo "Stepping on the gras..."
python train.py --model_type pix_shuffle --epochs 1 --batch_size 16 --accumulation_steps  1 --learning_rate 0.0005444628606687482 --checkpoint_interval 5  --generator_train_dir ../dataset_generator/dataset_lores --train_samples 50000 --val_samples 5000 --val_split_ratio 0.1 --crop_size 376 288 --checkpoint_dir model_conv6_run9 --early_stopping_patience 10
echo "Digging a hole in the landscape..."
python train.py --model_type pix_shuffle --epochs 50 --batch_size 32 --accumulation_steps  1 --learning_rate 0.0009908221381211726 --checkpoint_interval 5  --generator_train_dir ../dataset_generator/dataset_lores --train_samples 50000 --val_samples 5000 --val_split_ratio 0.1 --crop_size 376 288 --checkpoint_dir model_conv6_run9 --early_stopping_patience 10

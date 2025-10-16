#!/bin/bash

# 1. Input Check
if [ -z "$1" ]; then
    echo "Error: No model type specified."
    echo "Usage: $0 <model_type> [epochs_for_digging]"
    exit 1
fi

EPOCHS_DIGGING=${2:-16}

# 2. Set initial batch size and accumulation steps based on model type
if [ "$1" = "large" ]; then
    BATCH_SIZE=16
    ACCUMULATION_STEPS=4
else
    BATCH_SIZE=30
    ACCUMULATION_STEPS=2
fi

echo "Stepping on the gras..."
python train.py --model_type $1 --epochs 1 --batch_size $BATCH_SIZE --learning_rate 0.0004 --generator_train_dir ../dataset_generator/dataset --train_samples 100000 --val_samples 10000 --val_split_ratio 0.1 --crop_size 368 288 --checkpoint_dir $1 --identity_percentage 0.1

---

# 3. Apply new rule for "Digging" phase
# If epochs is more than 50, set accumulation to 1 and keep batchsize to 30.
if [ $EPOCHS_DIGGING -gt 50 ]; then
    # When this condition is met, BATCH_SIZE must be 30 and ACCUMULATION_STEPS must be 1.
    # The 'else' branch of the initial check already sets BATCH_SIZE=30, 
    # but we must ensure it's 30 and override ACCUMULATION_STEPS.
    BATCH_SIZE=30 
    ACCUMULATION_STEPS=1
    echo "INFO: Epochs ($EPOCHS_DIGGING) > 50. Overriding BATCH_SIZE to $BATCH_SIZE and ACCUMULATION_STEPS to $ACCUMULATION_STEPS for 'Digging' phase."
fi

echo "Digging a hole in the landscape..."
python train.py --model_type $1 --epochs $EPOCHS_DIGGING --batch_size $BATCH_SIZE --accumulation_steps $ACCUMULATION_STEPS --learning_rate 0.0004 --checkpoint_interval 1 --generator_train_dir ../dataset_generator/dataset --train_samples 100000 --val_samples 10000 --val_split_ratio 0.1 --crop_size 368 288 --checkpoint_dir $1 --early_stopping_patience 30 --identity_percentage 0.01

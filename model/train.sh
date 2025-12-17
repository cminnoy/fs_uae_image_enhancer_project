#!/bin/bash

# Input Check
if [ -z "$1" ]; then
    echo "Error: No model type specified."
    echo "Usage: $0 <model_type> [epochs_for_digging]"
    exit 1
fi

MODEL_TYPE="$1"
EPOCHS_DIGGING="${2:-16}"

# Set initial batch size based on model type
if [ "$MODEL_TYPE" = "light" ]; then
    BATCH_SIZE=128
    echo "Using LIGHT model with batch size $BATCH_SIZE"
else
    BATCH_SIZE=64
    echo "Using FULL model with batch size $BATCH_SIZE"
fi

DATASET="../dataset_generator/dataset/dataset_train_ocs_games"
CHECKPOINT="$MODEL_TYPE/best_model.pth"

# --------------------------------------------------
# Phase 1: "Stepping on the grass" (ONLY if no checkpoint exists)
# --------------------------------------------------
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "No existing checkpoint found. Performing initial training step..."
    torchrun --nproc_per_node 1 train.py \
        --model_type "$MODEL_TYPE" \
        --epochs 1 \
        --batch_size "$BATCH_SIZE" \
        --learning_rate 0.0004 \
        --data_dir "$DATASET" \
        --shuffle_data \
        --samples_per_epoch 3300 \
        --checkpoint_dir "$MODEL_TYPE" \
        --num_workers 8 \
        --use_amp
else
    echo "Checkpoint exists ($CHECKPOINT). Skipping initial training step."
fi

# --------------------------------------------------
# Phase 2: "Digging a hole in the landscape" (always runs)
# --------------------------------------------------
torchrun --nproc_per_node 2 train.py \
    --load_checkpoint "$CHECKPOINT" \
    --model_type "$MODEL_TYPE" \
    --epochs "$EPOCHS_DIGGING" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 0.0004 \
    --data_dir "$DATASET" \
    --samples_per_epoch 33000 \
    --checkpoint_dir "$MODEL_TYPE" \
    --num_workers 12 \
    --use_amp


#!/bin/bash

# Input Check
if [ -z "$1" ]; then
    echo "Error: No model type specified."
    echo "Usage: $0 <model_type> [epochs_for_digging]"
    exit 1
fi

MODEL_TYPE="$1"
EPOCHS_DIGGING="${2:-16}"
if [ -z "${3:-}" ]; then
    TRIAL=""
else
    TRIAL="_${3}"
fi

# Set initial batch size based on model type
if [ "$MODEL_TYPE" = "light" ]; then
    BATCH_SIZE=12
    LEARNING_RATE=0.0007
    WARMUP_STEPS=1000
    echo "Using Mamba LIGHT model with batch size $BATCH_SIZE"
else
    BATCH_SIZE=8
    LEARNING_RATE=0.00045
    WARMUP_STEPS=2000
    echo "Using Mamba FULL model with batch size $BATCH_SIZE"
fi

DATASET1="../dataset_generator/dataset/dataset_ocs/train"
DATASET2="../dataset_generator/dataset2/dataset_ocs/train"
MODEL_DIR="${MODEL_TYPE}_ocs_games${TRIAL}"

# Prefer the latest epoch checkpoint (highest epoch number). If none, fall back to best_model.pth.
LATEST_EPOCH_CHECKPOINT=$(ls -1 "${MODEL_DIR}"/epoch_*.pth 2>/dev/null | sort -V | tail -n 1)
if [ -n "$LATEST_EPOCH_CHECKPOINT" ]; then
    CHECKPOINT="$LATEST_EPOCH_CHECKPOINT"
elif [ -f "${MODEL_DIR}/best_model.pth" ]; then
    CHECKPOINT="${MODEL_DIR}/best_model.pth"
else
    CHECKPOINT=""
fi

SAMPLES_PER_EPOCH=46000
SAMPLES_VALIDATION=4600

# --------------------------------------------------
# Phase 1: "Stepping on the grass" (ONLY if no checkpoint exists)
# --------------------------------------------------
if [ -z "$CHECKPOINT" ] || [[ ! -f "$CHECKPOINT" ]]; then
    echo "No existing checkpoint found. Performing initial training step..."
    torchrun --nproc_per_node 2 train.py \
        --model_type "$MODEL_TYPE" \
        --epochs 1 \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --warmup_steps "$WARMUP_STEPS" \
        --data_dir "$DATASET1" "$DATASET2" \
        --generator_crop_size "752 576" \
        --train_crop_size "752 576" \
        --shuffle_data \
        --samples_per_epoch $SAMPLES_PER_EPOCH \
        --lores_only \
        --checkpoint_dir "$MODEL_DIR" \
        --num_workers 8 \
        --use_amp \
        --log_dir "runs/ocs_games/${MODEL_TYPE}${TRIAL}" \
        --val_limit $SAMPLES_VALIDATION
else
    echo "Found checkpoint ($CHECKPOINT). Skipping initial training step."
fi

# Recompute latest checkpoint after Phase 1 in case Phase 1 created new checkpoints
LATEST_EPOCH_CHECKPOINT=$(ls -1 "${MODEL_DIR}"/epoch_*.pth 2>/dev/null | sort -V | tail -n 1)
if [ -n "$LATEST_EPOCH_CHECKPOINT" ]; then
    CHECKPOINT="$LATEST_EPOCH_CHECKPOINT"
elif [ -f "${MODEL_DIR}/best_model.pth" ]; then
    CHECKPOINT="${MODEL_DIR}/best_model.pth"
else
    CHECKPOINT=""
fi

# --------------------------------------------------
# Phase 2: "Digging a hole in the landscape" (always runs)
# --------------------------------------------------
if [ -n "$CHECKPOINT" ] && [[ -f "$CHECKPOINT" ]]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    torchrun --nproc_per_node 2 train.py \
        --load_checkpoint "$CHECKPOINT" \
        --model_type "$MODEL_TYPE" \
        --epochs "$EPOCHS_DIGGING" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --warmup_steps "$WARMUP_STEPS" \
        --data_dir "$DATASET1" "$DATASET2" \
        --generator_crop_size "752 576" \
        --train_crop_size "752 576" \
        --samples_per_epoch $SAMPLES_PER_EPOCH \
        --checkpoint_dir "$MODEL_DIR" \
        --lores_only \
        --num_workers 8 \
        --use_amp \
        --log_dir "runs/ocs_games/${MODEL_TYPE}${TRIAL}" \
        --val_limit $SAMPLES_VALIDATION \
        --early-stop-patience 30 
else
    echo "No checkpoint to load — starting without --load_checkpoint"
    torchrun --nproc_per_node 2 train.py \
        --model_type "$MODEL_TYPE" \
        --epochs "$EPOCHS_DIGGING" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --warmup_steps "$WARMUP_STEPS" \
        --data_dir "$DATASET1" "$DATASET2" \
        --generator_crop_size "752 576" \
        --train_crop_size "752 576" \
        --samples_per_epoch $SAMPLES_PER_EPOCH \
        --checkpoint_dir "$MODEL_DIR" \
        --lores_only \
        --num_workers 8 \
        --use_amp \
        --log_dir "runs/ocs_games/${MODEL_TYPE}${TRIAL}" \
        --val_limit $SAMPLES_VALIDATION \
        --early-stop-patience 30
fi


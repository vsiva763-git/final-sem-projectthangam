#!/bin/bash

# Train speech enhancement model using Kaggle VoiceBank dataset
# This script uses the dataset downloaded from Kaggle

echo "=========================================="
echo "Speech Enhancement Model Training"
echo "Dataset: Kaggle VoiceBank"
echo "=========================================="
echo ""

# Check if dataset is available
DATASET_PATH="/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1"

if [ ! -d "$DATASET_PATH" ]; then
    echo "Downloading Kaggle dataset..."
    python download_dataset.py
    echo ""
fi

# Set training parameters
EPOCHS=${1:-20}
BATCH_SIZE=${2:-8}
NUM_WORKERS=${3:-2}

echo "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Workers: $NUM_WORKERS"
echo ""

# Start training
python train_kaggle.py \
    --config configs/default_config.yaml \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --checkpoint-dir ./kaggle_checkpoints \
    --log-dir ./kaggle_logs

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints: ./kaggle_checkpoints"
echo "Logs: ./kaggle_logs"
echo "=========================================="

# Training with Kaggle VoiceBank Dataset

This guide explains how to train the speech enhancement model using the Kaggle VoiceBank dataset.

## Dataset Overview

**Dataset:** [VoiceBank Clean/Test + ESC-50 (Cry, Baby, Dog)](https://www.kaggle.com/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog)

The dataset contains:
- **824 paired audio samples** (clean and noisy versions)
- Clean audio in: `clean_testset_wav/clean_testset_wav/`
- Noisy audio in: `noisy_dataset_wav/noisy_dataset_wav/`
- Audio format: 16 kHz WAV files
- Speaker IDs: p232, p257, etc.

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
# Train with default settings (20 epochs, batch size 8)
./train_with_kaggle.sh

# Train with custom epochs
./train_with_kaggle.sh 50

# Train with custom epochs and batch size
./train_with_kaggle.sh 50 16

# Train with custom epochs, batch size, and workers
./train_with_kaggle.sh 50 16 4
```

### Option 2: Manual Python Command

```bash
# Download dataset first (if not already downloaded)
python download_dataset.py

# Start training
python train_kaggle.py \
    --config configs/default_config.yaml \
    --epochs 50 \
    --batch-size 8 \
    --num-workers 2 \
    --checkpoint-dir ./kaggle_checkpoints \
    --log-dir ./kaggle_logs
```

## Training Configuration

The dataset paths are configured in [configs/default_config.yaml](configs/default_config.yaml):

```yaml
data:
  n_fft: 512
  hop_length: 256
  max_length: 32000
  clean_dir: "/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/clean_testset_wav/clean_testset_wav"
  noisy_dir: "/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/noisy_dataset_wav/noisy_dataset_wav"
```

### Key Parameters

- **n_fft:** FFT size for STFT (512)
- **hop_length:** Hop length for STFT (256)
- **max_length:** Maximum audio length in samples (32000 = 2 seconds at 16kHz)
- **epochs:** Number of training epochs
- **batch_size:** Batch size for training
- **learning_rate:** Learning rate (default: 1e-3)

## Training Process

The training script:

1. **Loads the dataset** - Automatically uses the downloaded Kaggle dataset
2. **Splits data** - 80% training, 20% validation
3. **Processes audio** - Converts to STFT representations
4. **Trains model** - Uses combined loss (complex value + magnitude)
5. **Saves checkpoints** - Best and latest models saved automatically
6. **Logs metrics** - Training/validation loss tracked

### Dataset Split

- **Training:** ~659 samples (80%)
- **Validation:** ~165 samples (20%)
- Random split with seed=42 for reproducibility

## Output Files

After training, you'll find:

```
kaggle_checkpoints/
├── checkpoint_epoch_0.pt
├── checkpoint_epoch_1.pt
├── ...
└── best_model.pt

kaggle_logs/
└── training_log_YYYYMMDD_HHMMSS.json
```

### Checkpoint Contents

Each checkpoint includes:
- Model weights
- Optimizer state
- Epoch number
- Loss value
- Timestamp

## Monitoring Training

### View Training Progress

```bash
# Watch the training output in real-time
tail -f kaggle_logs/training_log_*.json
```

### Check Loss Values

The training log contains:
- Epoch number
- Training loss (combined, CV, magnitude)
- Validation loss (combined, CV, magnitude)
- Learning rate
- Timestamp

## Advanced Usage

### Custom Dataset Paths

Override paths via command line:

```bash
python train_kaggle.py \
    --clean-dir /path/to/clean/audio \
    --noisy-dir /path/to/noisy/audio \
    --config configs/default_config.yaml
```

### Enable Mixed Precision Training

For faster training on GPU:

```bash
python train_kaggle.py \
    --config configs/default_config.yaml \
    --use-amp
```

### Adjust Learning Parameters

```bash
python train_kaggle.py \
    --config configs/default_config.yaml \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.0005
```

## System Requirements

### Minimum
- **RAM:** 8 GB
- **CPU:** 4 cores
- **Storage:** 2 GB for dataset + checkpoints

### Recommended
- **RAM:** 16 GB+
- **GPU:** NVIDIA GPU with 4GB+ VRAM (for faster training)
- **CPU:** 8+ cores
- **Storage:** 5 GB

## Expected Training Time

| Configuration | Time per Epoch | 50 Epochs |
|--------------|----------------|-----------|
| CPU (4 cores) | ~15 min | ~12.5 hours |
| GPU (T4) | ~2 min | ~1.7 hours |
| GPU (V100) | ~1 min | ~50 min |

*Times are approximate and depend on system specifications*

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size

```bash
python train_kaggle.py --batch-size 4
```

### Issue: Dataset Not Found

**Solution:** Download dataset manually

```bash
python download_dataset.py
```

### Issue: Slow Training on CPU

**Solution:** Reduce workers and batch size

```bash
python train_kaggle.py --batch-size 4 --num-workers 0
```

### Issue: CUDA Out of Memory

**Solution:** Enable mixed precision or reduce batch size

```bash
python train_kaggle.py --use-amp --batch-size 4
```

## Model Inference

After training, use the trained model for inference:

```bash
python inference.py \
    --checkpoint kaggle_checkpoints/best_model.pt \
    --input_audio path/to/noisy_audio.wav \
    --output_audio path/to/enhanced_audio.wav
```

## Dataset Statistics

- **Total samples:** 824
- **Training samples:** 659 (80%)
- **Validation samples:** 165 (20%)
- **Sample rate:** 16 kHz
- **Duration range:** Varies (truncated to max_length if specified)
- **Speakers:** Multiple (p232, p257, etc.)

## Next Steps

1. **Monitor training** - Watch loss values decrease
2. **Evaluate model** - Test on validation set
3. **Fine-tune** - Adjust hyperparameters if needed
4. **Deploy** - Use trained model in web app

## Additional Resources

- [Dataset on Kaggle](https://www.kaggle.com/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog)
- [Model Architecture](README.md)
- [Web Application](WEB_APP_GUIDE.md)
- [Project Documentation](PROJECT_SUMMARY.md)

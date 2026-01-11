# Kaggle Dataset Integration - Summary

## ✅ Successfully Completed

Your speech enhancement model is now configured to train with the Kaggle VoiceBank dataset!

## What Was Set Up

### 1. **Dataset Download** ✓
- Downloaded 824 paired audio samples from Kaggle
- Dataset: `voicebank-cleantest-esc-crybaby-dog`
- Location: `~/.cache/kagglehub/datasets/anupamupadhaya/...`

### 2. **Package Installation** ✓
- Installed `kagglehub` package
- Added to `requirements.txt`

### 3. **Training Scripts** ✓
Created three new files:
- **`download_dataset.py`** - Downloads and inspects the Kaggle dataset
- **`train_kaggle.py`** - Training script for Kaggle dataset
- **`train_with_kaggle.sh`** - Convenient bash script to start training

### 4. **Configuration** ✓
Updated [configs/default_config.yaml](configs/default_config.yaml) with dataset paths:
```yaml
data:
  clean_dir: "/home/codespace/.cache/kagglehub/.../clean_testset_wav"
  noisy_dir: "/home/codespace/.cache/kagglehub/.../noisy_dataset_wav"
```

### 5. **Documentation** ✓
Created [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md) with:
- Quick start instructions
- Training configuration details
- Troubleshooting guide
- Performance benchmarks

## Dataset Details

| Metric | Value |
|--------|-------|
| Total Samples | 824 |
| Training Split | 659 (80%) |
| Validation Split | 165 (20%) |
| Sample Rate | 16 kHz |
| Format | WAV |
| Speakers | Multiple (p232, p257, etc.) |

## How to Start Training

### Quick Start (Easiest)
```bash
./train_with_kaggle.sh
```

### With Custom Parameters
```bash
# 50 epochs, batch size 16
./train_with_kaggle.sh 50 16

# Or use Python directly
python train_kaggle.py --config configs/default_config.yaml --epochs 50
```

## Model Architecture

- **Type:** Speech Enhancement Network (U-Net style)
- **Parameters:** 741,730
- **Input:** Noisy audio (STFT representation)
- **Output:** Enhanced audio (STFT representation)
- **Loss:** Combined loss (complex value + magnitude)

## Expected Results

After training, you'll have:
```
kaggle_checkpoints/
├── checkpoint_epoch_0.pt
├── checkpoint_epoch_1.pt
├── ...
└── best_model.pt          ← Best performing model

kaggle_logs/
└── training_log_*.json    ← Training metrics
```

## Next Steps

1. **Start Training:**
   ```bash
   ./train_with_kaggle.sh 50 8  # 50 epochs, batch size 8
   ```

2. **Monitor Progress:**
   - Watch the terminal for training/validation loss
   - Lower loss = better performance

3. **Use Trained Model:**
   ```bash
   python inference.py \
       --checkpoint kaggle_checkpoints/best_model.pt \
       --input_audio noisy_audio.wav \
       --output_audio enhanced_audio.wav
   ```

4. **Integrate with Web App:**
   - Replace demo checkpoints with trained model
   - Update [app.py](app.py) to use new checkpoint

## Training Time Estimates

| System | Time per Epoch | 50 Epochs |
|--------|----------------|-----------|
| CPU (4 cores) | ~15 min | ~12.5 hours |
| GPU (T4) | ~2 min | ~1.7 hours |
| GPU (V100) | ~1 min | ~50 min |

## Troubleshooting

### Out of Memory?
Reduce batch size:
```bash
./train_with_kaggle.sh 50 4  # Batch size 4 instead of 8
```

### Training Too Slow?
- Use GPU if available
- Reduce number of workers: `--num-workers 0`
- Enable mixed precision: `--use-amp`

## Files Created

1. `download_dataset.py` - Dataset downloader
2. `train_kaggle.py` - Training script
3. `train_with_kaggle.sh` - Bash wrapper
4. `KAGGLE_TRAINING_GUIDE.md` - Detailed guide
5. `KAGGLE_DATASET_SUMMARY.md` - This file

## Support

For more details, see:
- [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md) - Complete training guide
- [README.md](README.md) - Project overview
- [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) - Web application guide

---

**Ready to train!** 🚀

Run: `./train_with_kaggle.sh` to get started!

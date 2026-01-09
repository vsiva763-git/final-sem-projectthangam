# 🎵 How to Remove Noise from Your Audio Files

## Quick Start - 3 Simple Steps

### Step 1: Upload Your Noisy Audio File
Place your audio file (`.wav` format) in the project directory or note its path.

### Step 2: Run the Denoising Script
```bash
python denoise_audio.py --input your_noisy_audio.wav
```

The enhanced audio will be saved as `your_noisy_audio_enhanced.wav`

### Step 3: Listen to the Result!
Your cleaned audio is ready to use.

---

## Detailed Usage

### Basic Usage
```bash
# Simple - automatically creates output file
python denoise_audio.py --input noisy.wav

# Specify custom output filename
python denoise_audio.py --input noisy.wav --output clean.wav
```

### Advanced Options
```bash
# Use a different model checkpoint
python denoise_audio.py --input noisy.wav --model demo_checkpoints/checkpoint_epoch_1.pt

# Use GPU for faster processing (if available)
python denoise_audio.py --input noisy.wav --device cuda
```

---

## Supported Audio Formats

✅ **Supported:**
- WAV files (.wav)
- Mono or stereo (automatically converted to mono)
- Any sample rate (16kHz recommended)

⚠️ **For other formats (MP3, FLAC, etc.):**
Convert to WAV first using:
```bash
# Using ffmpeg
ffmpeg -i your_audio.mp3 your_audio.wav
```

---

## Example Commands

### Example 1: Basic Denoising
```bash
python denoise_audio.py --input my_recording.wav
# Output: my_recording_enhanced.wav
```

### Example 2: Batch Processing
```bash
# Process multiple files
for file in *.wav; do
    python denoise_audio.py --input "$file" --output "clean_$file"
done
```

### Example 3: With Custom Model
If you've trained your own model:
```bash
python denoise_audio.py \
    --input noisy_speech.wav \
    --output enhanced_speech.wav \
    --model checkpoints/best_model.pt
```

---

## What the Model Does

The speech enhancement model:
- ✅ Removes background noise
- ✅ Preserves speech quality
- ✅ Handles various noise types (white noise, babble, etc.)
- ✅ Works on mono-channel audio

### Processing Pipeline:
1. **Input:** Your noisy audio file
2. **STFT:** Converts audio to frequency domain
3. **Neural Network:** Separates speech from noise
4. **ISTFT:** Converts back to time domain
5. **Output:** Clean audio file

---

## Current Model Status

📌 **Available Models:**
- `demo_checkpoints/checkpoint_epoch_1.pt` - Demo model (small, fast)
- Ready to use immediately!

🎓 **For Better Results:**
Train a full model on your specific audio:
```bash
python train.py \
    --clean-dir ./data/clean \
    --noisy-dir ./data/noisy \
    --epochs 50 \
    --batch-size 8
```

---

## Troubleshooting

### ❌ "Input file not found"
Make sure the audio file path is correct:
```bash
# Use absolute path
python denoise_audio.py --input /full/path/to/audio.wav

# Or relative path from project directory
python denoise_audio.py --input data/audio.wav
```

### ❌ "Model checkpoint not found"
The demo model should be at `demo_checkpoints/checkpoint_epoch_1.pt`

If missing, run the examples first:
```bash
python examples.py
```

### ❌ Audio sounds distorted
The demo model is small and undertrained. For production use:
1. Collect clean speech data
2. Train a full-size model
3. Use the trained model for inference

---

## Performance Tips

### For Faster Processing:
- Use GPU: `--device cuda`
- Process shorter audio clips
- Use a smaller model

### For Better Quality:
- Train on domain-specific data
- Use larger base_channels (32 or 64)
- Train for more epochs (50-100)
- Use higher quality input audio (16kHz+)

---

## File Upload Options

### Option 1: Direct Upload (If using web interface)
Simply drag and drop your `.wav` file into the project directory

### Option 2: Command Line
```bash
# Copy file to project
cp /path/to/your/audio.wav /workspaces/final-sem-project/

# Then denoise
python denoise_audio.py --input audio.wav
```

### Option 3: Remote URL
```bash
# Download first
wget https://example.com/noisy_audio.wav

# Then denoise
python denoise_audio.py --input noisy_audio.wav
```

---

## Output Information

The script provides:
- ✅ Input audio duration and sample rate
- ✅ Processing progress
- ✅ Output file location
- ✅ Audio statistics (power levels, noise reduction)

Example output:
```
============================================================
🎵 Speech Enhancement - Noise Removal
============================================================

📂 Loading audio: noisy.wav
   Sample rate: 16000 Hz
   Duration: 5.23 seconds
   Samples: 83680

🔧 Processing audio...
   Computing STFT...
   STFT shape: torch.Size([1, 1, 257, 328])

🤖 Loading model: demo_checkpoints/checkpoint_epoch_1.pt
   Checkpoint from epoch: 1
   Running noise reduction...
   Computing inverse STFT...

💾 Saving enhanced audio: noisy_enhanced.wav
   Output duration: 5.24 seconds
   Output samples: 83968

============================================================
✅ Noise removal completed successfully!
============================================================
```

---

## Next Steps

1. **Try it now:** Upload your audio and run the script
2. **Train custom model:** Use your own data for better results
3. **Integrate:** Use the Python API in your applications
4. **Contribute:** Improve the model architecture

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `python denoise_audio.py --input file.wav` | Basic denoising |
| `python denoise_audio.py --input file.wav --output clean.wav` | Custom output |
| `python denoise_audio.py --help` | Show all options |
| `python examples.py` | Run all demos |
| `python train.py` | Train custom model |

---

**Ready to clean your audio? Just run:**
```bash
python denoise_audio.py --input YOUR_FILE.wav
```

Your enhanced audio will be saved automatically! 🎉

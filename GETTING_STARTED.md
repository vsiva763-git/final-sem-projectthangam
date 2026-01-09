# Getting Started Guide - Speech Enhancement Project

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Quick Examples
```bash
python examples.py
```

This will demonstrate:
- Model architecture
- Loss functions
- Data processing (STFT/ISTFT)
- Dataset creation
- Mini training loop

### 3. Try Inference
```bash
# Create a test audio file first, then:
python inference.py --model ./demo_checkpoints/best_model.pt --input-file test.wav --output-file enhanced.wav
```

## Step-by-Step Training Guide

### Step 1: Prepare Your Data

Create the following directory structure:
```
data/
├── clean/
│   ├── speech1.wav
│   ├── speech2.wav
│   └── ...
├── noisy/
│   ├── speech1.wav
│   ├── speech2.wav
│   └── ...
└── test/
    ├── test1.wav
    └── ...
```

**Or** use synthetic data generation (recommended for testing):
```bash
python train.py --clean-dir ./data/clean --train-samples 100 --val-samples 20
```

### Step 2: Configure Training

Edit `configs/default_config.yaml` or create a custom config:

```yaml
model:
  base_channels: 32

training:
  epochs: 100
  batch_size: 8
  learning_rate: 1e-3
  weight_decay: 1e-5

loss:
  alpha: 0.3
  lambda_cv: 0.5
  lambda_mag: 0.5

data:
  n_fft: 512
  hop_length: 256
  max_length: 32000
```

### Step 3: Train the Model

**Basic training:**
```bash
python train.py
```

**With custom configuration:**
```bash
python train.py --config configs/default_config.yaml --epochs 50 --batch-size 16
```

**With GPU acceleration and mixed precision:**
```bash
python train.py --use-amp --batch-size 32 --num-workers 4
```

### Step 4: Monitor Training

Training logs are saved to `./logs/` with metrics tracked over time.

Checkpoints are saved to `./checkpoints/`:
- `best_model.pt` - Best model based on validation loss
- `checkpoint_epoch_*.pt` - Checkpoints for each epoch

### Step 5: Evaluate and Enhance

**Enhance a single file:**
```bash
python inference.py \
    --model ./checkpoints/best_model.pt \
    --input-file noisy_sample.wav \
    --output-file enhanced_sample.wav \
    --compare
```

**Batch process directory:**
```bash
python inference.py \
    --model ./checkpoints/best_model.pt \
    --input-dir ./data/test \
    --output-dir ./data/enhanced
```

## Understanding the Architecture

### Data Flow

```
Audio Signal (WAV)
    ↓
STFT → [Real Part, Imaginary Part]
    ↓
Model Processing:
  ├─ Encoder (4 UDU layers, downsampling)
  ├─ Intermediate (2D-Conv + TSCAC + GWPS)
  └─ Twin Decoders (RUDU + IUDU, upsampling)
    ↓
ISTFT ← [Enhanced Real, Enhanced Imaginary]
    ↓
Enhanced Audio Signal (WAV)
```

### Key Components

**Unified Discovery Unit (UDU)**
- Uses two parallel convolutional pathways
- Different kernel sizes (3×3 and 5×5) for multi-scale feature extraction
- Gating mechanism balances contribution from both pathways

**Attention Mechanisms**
- Temporal Attention: Models temporal dependencies
- Spectral Attention: Models frequency dependencies
- Combined in TSCAC for comprehensive feature modeling

**Loss Functions**
- Complex Value Loss: Compresses and penalizes errors in real/imaginary parts
- Magnitude Loss: Penalizes errors in spectral magnitude
- Combined with weights λ_cv and λ_mag

## Advanced Usage

### Custom Dataset

```python
from torch.utils.data import Dataset
from speech_enhancement.data_processing import STFTProcessor

class MyDataset(Dataset):
    def __init__(self, noisy_files, clean_files):
        self.stft = STFTProcessor(n_fft=512, hop_length=256)
        self.noisy_files = noisy_files
        self.clean_files = clean_files
    
    def __getitem__(self, idx):
        import soundfile as sf
        import torch
        
        noisy_audio, _ = sf.read(self.noisy_files[idx])
        clean_audio, _ = sf.read(self.clean_files[idx])
        
        noisy_real, noisy_imag = self.stft.stft(noisy_audio)
        clean_real, clean_imag = self.stft.stft(clean_audio)
        
        return (
            noisy_real.unsqueeze(0),
            noisy_imag.unsqueeze(0),
            clean_real.unsqueeze(0),
            clean_imag.unsqueeze(0),
        )
```

### Custom Training Loop

```python
import torch
from speech_enhancement import SpeechEnhancementNetwork, CombinedLoss

model = SpeechEnhancementNetwork()
loss_fn = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters())
device = torch.device('cuda')

model = model.to(device)

for epoch in range(num_epochs):
    for batch in train_loader:
        noisy_real, noisy_imag, clean_real, clean_imag = [
            b.to(device) for b in batch
        ]
        
        # Forward pass
        enhanced_real, enhanced_imag = model(noisy_real, noisy_imag)
        
        # Compute loss
        loss, loss_dict = loss_fn(
            enhanced_real, enhanced_imag,
            clean_real, clean_imag
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Real-Time Inference

```python
import torch
import numpy as np
from speech_enhancement import SpeechEnhancementNetwork
from speech_enhancement.data_processing import STFTProcessor

class RealtimeEnhancer:
    def __init__(self, model_path, chunk_size=512):
        self.model = SpeechEnhancementNetwork()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.stft = STFTProcessor(n_fft=512, hop_length=256)
        self.chunk_size = chunk_size
        self.overlap_buffer = None
    
    def process_chunk(self, audio_chunk):
        """Process single audio chunk."""
        with torch.no_grad():
            real, imag = self.stft.stft(audio_chunk)
            real = real.unsqueeze(0).unsqueeze(0)
            imag = imag.unsqueeze(0).unsqueeze(0)
            
            enh_real, enh_imag = self.model(real, imag)
            
            enhanced = self.stft.istft(
                enh_real.squeeze(),
                enh_imag.squeeze()
            )
        
        return enhanced.numpy()
```

## Troubleshooting

### Problem: CUDA Out of Memory
**Solution:**
```bash
python train.py --batch-size 4  # Reduce batch size
python train.py --use-amp       # Enable mixed precision
```

### Problem: Slow Training
**Solution:**
```bash
python train.py --num-workers 4 --use-amp  # Parallel loading + mixed precision
```

### Problem: Poor Enhancement Quality
**Check:**
1. Audio sample rates match
2. Noise characteristics similar to training
3. Model trained for enough epochs
4. Validation loss decreasing

### Problem: STFT Mismatch
**Ensure consistency:**
```python
# Training
train_stft = STFTProcessor(n_fft=512, hop_length=256)

# Inference (must match!)
inference_stft = STFTProcessor(n_fft=512, hop_length=256)
```

## Performance Optimization

### For Training
1. Use mixed precision: `--use-amp`
2. Increase workers: `--num-workers 4`
3. Larger batches if memory allows
4. Use GPU: `torch.cuda.is_available()`

### For Inference
1. Use batch processing
2. Reduce audio length if needed
3. Use half-precision (FP16) on GPU
4. Convert to TorchScript for deployment

### Inference Speed Tips
```python
# Enable evaluation mode
model.eval()

# Use torch.no_grad()
with torch.no_grad():
    output = model(input)

# Use half precision
model.half()
audio = audio.half()

# Or convert to TorchScript
traced = torch.jit.trace(model, (real, imag))
torch.jit.save(traced, 'model.pt')
```

## Evaluation Metrics

Install additional packages for evaluation:
```bash
pip install pesq pystoi
```

Use standard metrics:
```python
from pesq import pesq
from pystoi import stoi

# PESQ: Perceptual Evaluation of Speech Quality (0-4.5, higher is better)
pesq_score = pesq(sample_rate, clean_audio, enhanced_audio, 'wb')

# STOI: Short-Time Objective Intelligibility (0-1, higher is better)
stoi_score = stoi(clean_audio, enhanced_audio, sample_rate)

# SNR: Signal-to-Noise Ratio (dB, higher is better)
def snr_db(clean, enhanced):
    return 10 * np.log10(np.sum(clean**2) / np.sum((clean - enhanced)**2))
```

## Common Parameters and Their Effects

| Parameter | Effect | Recommended |
|-----------|--------|------------|
| base_channels | Network size (memory, capacity) | 32-64 |
| n_fft | FFT resolution (freq/time tradeoff) | 512 |
| hop_length | Overlap amount (timing resolution) | n_fft/2 |
| learning_rate | Convergence speed | 1e-3 to 1e-4 |
| batch_size | Training stability | 8-32 |
| alpha (compression) | Spectral compression | 0.3-0.5 |
| lambda_cv | Complex loss weight | 0.5 |
| lambda_mag | Magnitude loss weight | 0.5 |

## File Formats

### Input Audio
- Format: WAV (PCM)
- Channels: Mono or Stereo (auto-converted)
- Sample Rate: Any (matched to training)
- Bit Depth: Any (auto-normalized)

### Output Audio
- Format: WAV (floating point)
- Channels: Mono
- Sample Rate: Same as input
- Bit Depth: 32-bit float

## Deployment

### Export Model
```python
import torch
model = torch.load('best_model.pt')
scripted = torch.jit.script(model)
torch.jit.save(scripted, 'model_deployment.pt')
```

### Use in Production
```python
device = torch.device('cpu')  # or 'cuda'
model = torch.jit.load('model_deployment.pt')
model = model.to(device)

# Inference
with torch.no_grad():
    output = model(input)
```

## Next Steps

1. **Start Simple**: Run `examples.py` first
2. **Prepare Data**: Gather your audio files
3. **Train**: Use `train.py` with synthetic or real data
4. **Evaluate**: Check model checkpoint quality
5. **Deploy**: Use `inference.py` or integrate into your application
6. **Optimize**: Fine-tune parameters based on results
7. **Experiment**: Try custom architectures and configurations

## Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- Speech Enhancement Papers: arXiv (search "speech enhancement")
- Audio Processing: https://librosa.org/
- Training Best Practices: https://pytorch.org/tutorials/

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review code comments and docstrings
3. Run examples to verify setup
4. Check configuration parameters

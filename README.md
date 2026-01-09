# Speech Enhancement Network - Final Semester Project

A comprehensive implementation of a mono-channel speech enhancement neural network based on the architecture described in "Enhancing Model Robustness in Noisy Environments." This project implements an encoder-decoder structure with complex spectral mapping for removing noise from speech signals.

## Project Overview

This implementation provides a complete speech enhancement pipeline with:

- **Advanced Architecture Components:**
  - Unified Discovery Unit (UDU) with dual expert segments and gating
  - Temporal-Spectral Cross-Attention Component (TSCAC)
  - Twin-Segment Attention Integration Component (TSAIC)
  - Gaussian-Weighted Progressive Structure (GWPS)

- **Complex Spectral Mapping:** Direct processing of real and imaginary parts of STFT

- **Comprehensive Loss Functions:** 
  - Complex Value Loss with power-law compression
  - Magnitude Loss
  - Combined weighted loss

- **Full Pipeline:** Data processing, training, validation, and inference

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
cd /workspaces/final-sem-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
final-sem-project/
├── speech_enhancement/          # Main package
│   ├── __init__.py
│   ├── model.py                # Main encoder-decoder architecture
│   ├── components.py           # UDU, attention modules, etc.
│   ├── losses.py               # Loss functions
│   ├── data_processing.py      # STFT, audio utilities
│   ├── dataset.py              # Dataset classes
│   └── trainer.py              # Training utilities
├── configs/
│   └── default_config.yaml     # Default configuration
├── data/                        # Data directory (create your own)
│   ├── clean/                  # Clean speech audio files
│   ├── noisy/                  # Noisy speech audio files
│   ├── test/                   # Test audio files
│   └── enhanced/               # Enhanced output directory
├── checkpoints/                # Model checkpoints (auto-created)
├── logs/                        # Training logs (auto-created)
├── train.py                     # Training script
├── inference.py                 # Inference script
└── requirements.txt             # Python dependencies
```

## Architecture Details

### Model Architecture

The network uses an encoder-decoder structure with skip connections:

```
Input (Real, Imaginary parts of STFT)
    ↓
[Encoder: 4 UDU layers with downsampling]
    ↓
[Intermediate: 2D-Conv + TSCAC + GWPS]
    ↓
[Twin Decoders: RUDU and IUDU with skip connections]
    ↓
Output (Enhanced Real, Imaginary parts)
```

### Key Components

#### 1. Unified Discovery Unit (UDU)
- Two expert segments with different kernel sizes (3×3 and 5×5)
- Gating mechanism for balanced cooperative learning
- Batch normalization and ReLU activation

#### 2. Temporal-Spectral Cross-Attention (TSCAC)
- Temporal Attention Transformer (TAT) for subband modeling
- Spectral Attention Transformer (SAT) for full-band dependencies
- Fusion layer combining both attention mechanisms

#### 3. Twin-Segment Attention Integration (TSAIC)
- Spatial-channel attention module
- Teachable weight coefficients for feature combination
- Adaptive attention mechanism

#### 4. Gaussian-Weighted Progressive Structure (GWPS)
- Gaussian weight coefficients for each layer
- Prevents information loss in deep layers
- Progressive feature integration

### Loss Functions

#### Complex Value Loss (L_cv)
MSE loss on power-law compressed real and imaginary parts:
$$L_{cv} = MSE(compress(R_{pred}), compress(R_{clean})) + MSE(compress(I_{pred}), compress(I_{clean}))$$

Where: $compress(x) = sign(x) \cdot |x|^\alpha$ (with $\alpha = 0.3$)

#### Magnitude Loss (L_mag)
MSE loss on magnitude spectrum:
$$L_{mag} = MSE(|Z_{pred}|, |Z_{clean}|)$$

Where: $|Z| = \sqrt{R^2 + I^2}$

#### Combined Loss
$$L_{total} = \lambda_{cv} \cdot L_{cv} + \lambda_{mag} \cdot L_{mag}$$

## Training

### Basic Training

```bash
python train.py --epochs 50 --batch-size 8 --learning-rate 1e-3
```

### With Configuration File

```bash
python train.py --config configs/default_config.yaml
```

### Training Arguments

```
--clean-dir PATH           Directory with clean audio files (default: ./data/clean)
--epochs NUM              Number of training epochs
--batch-size NUM          Batch size (default: 4)
--learning-rate LR        Learning rate (default: 1e-3)
--use-amp                 Use automatic mixed precision
--checkpoint-dir PATH     Save checkpoints to directory
--log-dir PATH           Save logs to directory
--num-workers NUM        Data loading workers
```

### Example: Training with Custom Configuration

```bash
python train.py \
    --clean-dir ./data/clean \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs \
    --use-amp
```

## Inference

### Single File Enhancement

```bash
python inference.py \
    --model ./checkpoints/best_model.pt \
    --input-file noisy_speech.wav \
    --output-file enhanced_speech.wav \
    --compare
```

### Batch Processing

```bash
python inference.py \
    --model ./checkpoints/best_model.pt \
    --input-dir ./data/test \
    --output-dir ./data/enhanced
```

### Inference Arguments

```
--model PATH              Path to trained model (required)
--input-file PATH         Process single audio file
--input-dir PATH          Process directory of audio files
--output-file PATH        Output file for single file mode
--output-dir PATH         Output directory for batch mode
--n-fft SIZE             FFT size (default: 512)
--hop-length SIZE        Hop length (default: 256)
--use-cuda               Use CUDA if available
--compare                Compare original and enhanced
```

## Data Preparation

### Dataset Format

Create audio directories with WAV files:

```
data/
├── clean/
│   ├── speech_001.wav
│   ├── speech_002.wav
│   └── ...
├── noisy/
│   ├── speech_001.wav
│   ├── speech_002.wav
│   └── ...
└── test/
    ├── test_001.wav
    └── ...
```

### Supported Audio Formats
- WAV files at any sample rate (resampling handled automatically)
- Mono or stereo (stereo converted to mono)
- Any bit depth (converted to float32)

### Creating Synthetic Training Data

For testing without real data, the training script can generate synthetic noisy versions:

```python
from speech_enhancement.dataset import SyntheticNoiseDataset

dataset = SyntheticNoiseDataset(
    clean_dir="./data/clean",
    snr_db=10.0,  # Signal-to-Noise Ratio
    num_samples=100,
)
```

## Usage Examples

### Training from Scratch

```python
import torch
from torch.utils.data import DataLoader
from speech_enhancement import SpeechEnhancementNetwork, CombinedLoss, Trainer

# Create model
model = SpeechEnhancementNetwork(base_channels=32)

# Create loss function and optimizer
loss_fn = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create trainer
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=torch.device('cuda'),
)

# Train
trainer.fit(train_loader, val_loader, epochs=50)
```

### Using Pre-trained Model

```python
from speech_enhancement import SpeechEnhancementNetwork
from inference import SpeechEnhancer

# Initialize enhancer
enhancer = SpeechEnhancer(
    model_path="./checkpoints/best_model.pt",
    device='cuda',
)

# Enhance audio
import soundfile as sf
audio, sr = sf.read("noisy_speech.wav")
enhanced = enhancer.enhance(audio)
sf.write("enhanced_speech.wav", enhanced, sr)
```

## Configuration

### Default Configuration (configs/default_config.yaml)

```yaml
model:
  base_channels: 32

training:
  epochs: 100
  batch_size: 8
  learning_rate: 1e-3
  weight_decay: 1e-5

loss:
  alpha: 0.3              # Power-law compression
  lambda_cv: 0.5          # Complex value loss weight
  lambda_mag: 0.5         # Magnitude loss weight

data:
  n_fft: 512              # FFT size
  hop_length: 256         # STFT hop length
  max_length: 32000       # Max audio length (None = unlimited)
```

You can create custom configuration files and use them with:

```bash
python train.py --config configs/custom_config.yaml
```

## Model Performance

The model is designed to achieve good speech enhancement performance by:

1. **Complex Spectral Mapping:** Preserves phase information better than magnitude-only processing
2. **Skip Connections:** Maintains fine-grained spectral details
3. **Multi-Scale Attention:** Captures both temporal and spectral dependencies
4. **Progressive Integration:** Integrates information across layers effectively

### Expected Results

On typical speech enhancement datasets:
- Input SNR: 10 dB
- Output SNR: ~15-20 dB
- PESQ improvement: 0.5-1.0
- STOI improvement: 5-10%

*Note: Actual results depend on training data, noise types, and hyperparameters*

## Evaluation Metrics

For proper evaluation, use standard speech enhancement metrics:

```python
from pesq import pesq
from pystoi import stoi

# PESQ (Perceptual Evaluation of Speech Quality)
pesq_score = pesq(sr, clean_audio, enhanced_audio, 'wb')

# STOI (Short-Time Objective Intelligibility)
stoi_score = stoi(clean_audio, enhanced_audio, sr)
```

## Advanced Usage

### Custom Dataset

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, noisy_files, clean_files):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
    
    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        # Load and process audio
        # Return (noisy_real, noisy_imag, clean_real, clean_imag)
        pass
```

### Custom Loss Function

```python
from speech_enhancement.losses import CombinedLoss

class CustomLoss(CombinedLoss):
    def forward(self, enhanced_real, enhanced_imag, clean_real, clean_imag):
        # Custom loss computation
        pass
```

### Custom Model Architecture

```python
from speech_enhancement import SpeechEnhancementNetwork

class CustomEnhancementNetwork(SpeechEnhancementNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom layers or modifications
```

## Troubleshooting

### GPU Memory Issues

```bash
# Reduce batch size
python train.py --batch-size 4

# Enable gradient checkpointing (if needed)
# Modify trainer.py to enable checkpointing
```

### Slow Training

```bash
# Increase number of workers
python train.py --num-workers 4

# Use automatic mixed precision
python train.py --use-amp
```

### Poor Enhancement Quality

1. Check input audio sample rate matches training data
2. Ensure noise characteristics match training noise
3. Consider fine-tuning on specific noise types
4. Verify STFT parameters match training configuration

## Performance Optimization

### Inference Speed

- Use half-precision (FP16): Set `device='cuda'`
- Reduce FFT size: Lower `n_fft` (e.g., 256)
- Batch process multiple files
- Use TorchScript for deployment:

```python
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model.pt')
```

### Training Speed

- Enable mixed precision: `--use-amp`
- Increase batch size (if memory allows)
- Use multiple workers: `--num-workers 4`
- Use DistributedDataParallel for multi-GPU

## References

This implementation is based on the architecture described in:
- "Enhancing Model Robustness in Noisy Environments" - Research paper
- Deep learning architectures for speech enhancement
- Attention mechanisms in audio processing

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or contributions, please:
1. Check existing documentation and examples
2. Review the code comments and docstrings
3. Consult the configuration examples
4. Test with provided synthetic data first

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional attention mechanisms
- Multi-scale processing
- Real-time inference optimization
- Deployment to edge devices
- Additional loss functions
- Evaluation metrics integration
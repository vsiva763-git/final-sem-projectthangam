#!/bin/bash
# Verification Script for Speech Enhancement Project

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Speech Enhancement Project - Installation Verification      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "Checking Python version..."
python --version || echo "ERROR: Python not found"
echo ""

# Check Python packages
echo "Checking required packages..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch missing"
python -c "import torchaudio; print(f'✓ TorchAudio {torchaudio.__version__}')" || echo "✗ TorchAudio missing"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" || echo "✗ NumPy missing"
python -c "import scipy; print(f'✓ SciPy {scipy.__version__}')" || echo "✗ SciPy missing"
python -c "import soundfile; print('✓ SoundFile installed')" || echo "✗ SoundFile missing"
echo ""

# Check CUDA availability
echo "Checking GPU support..."
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')" || echo "✗ Error checking CUDA"
echo ""

# Check project structure
echo "Checking project structure..."
if [ -d "speech_enhancement" ]; then echo "✓ speech_enhancement/ directory exists"; else echo "✗ speech_enhancement/ missing"; fi
if [ -f "speech_enhancement/__init__.py" ]; then echo "✓ Package __init__.py exists"; else echo "✗ __init__.py missing"; fi
if [ -f "speech_enhancement/model.py" ]; then echo "✓ model.py exists"; else echo "✗ model.py missing"; fi
if [ -f "speech_enhancement/components.py" ]; then echo "✓ components.py exists"; else echo "✗ components.py missing"; fi
if [ -f "speech_enhancement/losses.py" ]; then echo "✓ losses.py exists"; else echo "✗ losses.py missing"; fi
if [ -f "speech_enhancement/data_processing.py" ]; then echo "✓ data_processing.py exists"; else echo "✗ data_processing.py missing"; fi
if [ -f "speech_enhancement/dataset.py" ]; then echo "✓ dataset.py exists"; else echo "✗ dataset.py missing"; fi
if [ -f "speech_enhancement/trainer.py" ]; then echo "✓ trainer.py exists"; else echo "✗ trainer.py missing"; fi
echo ""

# Check scripts
echo "Checking main scripts..."
if [ -f "train.py" ]; then echo "✓ train.py exists"; else echo "✗ train.py missing"; fi
if [ -f "inference.py" ]; then echo "✓ inference.py exists"; else echo "✗ inference.py missing"; fi
if [ -f "examples.py" ]; then echo "✓ examples.py exists"; else echo "✗ examples.py missing"; fi
echo ""

# Check documentation
echo "Checking documentation..."
if [ -f "README.md" ]; then echo "✓ README.md exists"; else echo "✗ README.md missing"; fi
if [ -f "GETTING_STARTED.md" ]; then echo "✓ GETTING_STARTED.md exists"; else echo "✗ GETTING_STARTED.md missing"; fi
if [ -f "PROJECT_SUMMARY.md" ]; then echo "✓ PROJECT_SUMMARY.md exists"; else echo "✗ PROJECT_SUMMARY.md missing"; fi
echo ""

# Try importing the package
echo "Testing package import..."
python -c "from speech_enhancement import SpeechEnhancementNetwork; print('✓ Successfully imported SpeechEnhancementNetwork')" || echo "✗ Failed to import SpeechEnhancementNetwork"
python -c "from speech_enhancement import CombinedLoss; print('✓ Successfully imported CombinedLoss')" || echo "✗ Failed to import CombinedLoss"
python -c "from speech_enhancement import Trainer; print('✓ Successfully imported Trainer')" || echo "✗ Failed to import Trainer"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Installation Verification Complete                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. Run examples:        python examples.py"
echo "2. Start training:      python train.py --epochs 10"
echo "3. Process audio:       python inference.py --help"
echo ""

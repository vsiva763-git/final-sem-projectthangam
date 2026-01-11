"""
Quick test of the training pipeline with reduced data
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from speech_enhancement.dataset import SpeechEnhancementDataset, FramePaddingCollate
from speech_enhancement.model import SpeechEnhancementNetwork

def test_training():
    """Quick test to verify training works."""
    
    print("=" * 70)
    print("Quick Training Test")
    print("=" * 70)
    
    # Paths
    clean_dir = "/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/clean_testset_wav/clean_testset_wav"
    noisy_dir = "/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/noisy_dataset_wav/noisy_dataset_wav"
    
    # Create dataset
    print("\n1. Loading dataset...")
    try:
        dataset = SpeechEnhancementDataset(
            noisy_dir=noisy_dir,
            clean_dir=clean_dir,
            n_fft=512,
            hop_length=256,
            max_length=32000,
        )
        print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return False
    
    # Create dataloader
    print("\n2. Creating dataloader...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=FramePaddingCollate(),
            num_workers=0,
        )
        print(f"   ✓ Dataloader created: {len(loader)} batches")
    except Exception as e:
        print(f"   ✗ Error creating dataloader: {e}")
        return False
    
    # Test batch loading
    print("\n3. Testing batch loading...")
    try:
        batch_iter = iter(loader)
        batch = next(batch_iter)
        noisy_real, noisy_imag, clean_real, clean_imag = batch
        print(f"   ✓ Batch loaded successfully")
        print(f"     - Noisy real shape: {noisy_real.shape}")
        print(f"     - Noisy imag shape: {noisy_imag.shape}")
        print(f"     - Clean real shape: {clean_real.shape}")
        print(f"     - Clean imag shape: {clean_imag.shape}")
    except Exception as e:
        print(f"   ✗ Error loading batch: {e}")
        return False
    
    # Create model
    print("\n4. Creating model...")
    try:
        model = SpeechEnhancementNetwork(base_channels=16)
        device = torch.device('cpu')
        model = model.to(device)
        print(f"   ✓ Model created")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"     - Total parameters: {total_params:,}")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        return False
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    try:
        with torch.no_grad():
            noisy_real = noisy_real.to(device)
            noisy_imag = noisy_imag.to(device)
            output = model(noisy_real, noisy_imag)
        print(f"   ✓ Forward pass successful")
        print(f"     - Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All tests passed! Training pipeline is working.")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)

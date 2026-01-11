"""
Simple test training without tqdm to find the issue
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from speech_enhancement.dataset import SpeechEnhancementDataset, FramePaddingCollate
from speech_enhancement.model import SpeechEnhancementNetwork
from speech_enhancement.losses import CombinedLoss

def simple_train_test():
    """Simple training test without tqdm."""
    
    print("=" * 70)
    print("Simple Training Test (Debugging)")
    print("=" * 70)
    
    # Paths
    clean_dir = "/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/clean_testset_wav/clean_testset_wav"
    noisy_dir = "/home/codespace/.cache/kagglehub/datasets/anupamupadhaya/voicebank-cleantest-esc-crybaby-dog/versions/1/noisy_dataset_wav/noisy_dataset_wav"
    
    # Create dataset
    print("\n1. Creating dataset...")
    dataset = SpeechEnhancementDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        n_fft=512,
        hop_length=256,
        max_length=32000,
    )
    print(f"   ✓ Dataset: {len(dataset)} samples")
    
    # Create small subset for testing
    print("\n2. Creating small test subset...")
    small_dataset, _ = torch.utils.data.random_split(
        dataset,
        [10, len(dataset) - 10],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"   ✓ Test dataset: {len(small_dataset)} samples")
    
    # Create dataloader
    print("\n3. Creating dataloader...")
    loader = DataLoader(
        small_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=FramePaddingCollate(),
        num_workers=0,
    )
    print(f"   ✓ Dataloader: {len(loader)} batches")
    
    # Create model
    print("\n4. Creating model...")
    device = torch.device('cpu')
    model = SpeechEnhancementNetwork(base_channels=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = CombinedLoss(alpha=0.3, lambda_cv=0.5, lambda_mag=0.5)
    print(f"   ✓ Model ready")
    
    # Training loop
    print("\n5. Starting training loop...")
    print("   " + "-" * 60)
    
    model.train()
    for epoch in range(1):
        total_loss = 0.0
        
        print(f"   Epoch {epoch + 1}/1")
        for batch_idx, batch in enumerate(loader):
            try:
                print(f"     Processing batch {batch_idx + 1}/{len(loader)}...", end=" ")
                
                # Unpack batch
                noisy_real, noisy_imag, clean_real, clean_imag = [
                    b.to(device) for b in batch
                ]
                print(f"shapes: {noisy_real.shape}", end=" ")
                
                # Forward pass
                start_time = time.time()
                enhanced_real, enhanced_imag = model(noisy_real, noisy_imag)
                forward_time = time.time() - start_time
                print(f"forward: {forward_time:.3f}s", end=" ")
                
                # Loss
                start_time = time.time()
                loss, loss_dict = loss_fn(
                    enhanced_real, enhanced_imag,
                    clean_real, clean_imag,
                )
                loss_time = time.time() - start_time
                print(f"loss: {loss_time:.3f}s", end=" ")
                
                # Backward
                start_time = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                backward_time = time.time() - start_time
                print(f"backward: {backward_time:.3f}s", end=" ")
                
                total_loss += loss.item()
                print(f"loss: {loss.item():.4f} ✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        avg_loss = total_loss / len(loader)
        print(f"   Epoch Loss: {avg_loss:.4f}")
    
    print("   " + "-" * 60)
    print("\n✓ Training test completed successfully!")
    return True

if __name__ == "__main__":
    success = simple_train_test()
    sys.exit(0 if success else 1)

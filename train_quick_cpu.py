"""
Quick CPU training script - small data, few epochs for testing
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
from pathlib import Path

from speech_enhancement.model import SpeechEnhancementNetwork
from speech_enhancement.losses import CombinedLoss
from speech_enhancement.trainer import Trainer
from speech_enhancement.dataset import SpeechEnhancementDataset, FramePaddingCollate


def main():
    print("=" * 60)
    print("QUICK CPU TRAINING - Testing Model")
    print("=" * 60)
    
    # Simple config
    config = {
        'model': {'base_channels': 16},
        'training': {
            'epochs': 3,  # Just 3 epochs for testing
            'batch_size': 2,  # Small batch
            'learning_rate': 0.0001,
            'weight_decay': 1e-5,
        },
        'loss': {'alpha': 0.3, 'lambda_cv': 0.5, 'lambda_mag': 0.5},
        'data': {'n_fft': 512, 'hop_length': 256, 'max_length': 16000},  # Shorter audio
    }
    
    # Use demo data
    clean_dir = "data/demo_clean"
    noisy_dir = "data/demo_clean"  # Use clean as noisy for testing
    
    if not Path(clean_dir).exists():
        print(f"❌ Demo data not found at {clean_dir}")
        print("   Creating synthetic data...")
        Path(clean_dir).mkdir(parents=True, exist_ok=True)
        
        # Create a simple test audio file
        import soundfile as sf
        import numpy as np
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(f"{clean_dir}/test_audio.wav", test_audio, 16000)
        print(f"   Created test audio: {clean_dir}/test_audio.wav")
    
    device = torch.device('cpu')
    print(f"\n✓ Device: {device}")
    
    # Create dataset
    print("\n📊 Loading data...")
    dataset = SpeechEnhancementDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        max_length=config['data']['max_length'],
    )
    
    if len(dataset) == 0:
        print("❌ No data found!")
        return
    
    print(f"   Total samples: {len(dataset)}")
    
    # Use small subset for quick training
    subset_size = min(10, len(dataset))
    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size
    
    indices = list(range(subset_size))
    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:])
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Data loaders
    collate_fn = FramePaddingCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Create model
    print("\n🤖 Creating model...")
    model = SpeechEnhancementNetwork(base_channels=config['model']['base_channels'])
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Loss and optimizer
    loss_fn = CombinedLoss(
        alpha=config['loss']['alpha'],
        lambda_cv=config['loss']['lambda_cv'],
        lambda_mag=config['loss']['lambda_mag'],
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir="./quick_checkpoints",
        log_dir="./quick_logs",
    )
    
    # Train
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        use_amp=False,  # No AMP on CPU
    )
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print("=" * 60)
    print(f"📁 Checkpoints: ./quick_checkpoints/")
    print(f"📊 Logs: ./quick_logs/")
    print("\nTest this model:")
    print("  cp quick_checkpoints/best_model.pt demo_checkpoints/working_model.pt")
    print("  python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

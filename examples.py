"""
Quick Start Example for Speech Enhancement
Demonstrates the complete workflow from training to inference.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Import speech enhancement components
from speech_enhancement import (
    SpeechEnhancementNetwork,
    CombinedLoss,
    Trainer,
    SyntheticNoiseDataset,
)
from speech_enhancement.dataset import FramePaddingCollate


def create_demo_audio(duration: float = 2.0, sr: int = 16000):
    """
    Create a simple demo audio file for testing.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        
    Returns:
        Audio array and sample rate
    """
    import soundfile as sf
    
    # Create synthetic speech-like signal
    t = np.linspace(0, duration, int(sr * duration))
    
    # Combination of sinusoids simulating speech
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
        0.15 * np.sin(2 * np.pi * 400 * t) +  # 1st harmonic
        0.1 * np.sin(2 * np.pi * 600 * t)    # 2nd harmonic
    )
    
    # Add some amplitude modulation for naturalness
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
    audio = audio * envelope
    
    return audio, sr


def example_1_basic_inference():
    """
    Example 1: Basic inference with a dummy model.
    Shows how to use the SpeechEnhancementNetwork.
    """
    print("=" * 60)
    print("Example 1: Basic Model Architecture")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model
    model = SpeechEnhancementNetwork(base_channels=32)
    model = model.to(device)
    
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    batch_size = 2
    freq_bins = 257  # For n_fft=512
    time_frames = 100
    
    real_part = torch.randn(batch_size, 1, freq_bins, time_frames).to(device)
    imag_part = torch.randn(batch_size, 1, freq_bins, time_frames).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Real part: {real_part.shape}")
    print(f"  Imag part: {imag_part.shape}")
    
    # Forward pass
    with torch.no_grad():
        enhanced_real, enhanced_imag = model(real_part, imag_part)
    
    print(f"\nOutput shapes:")
    print(f"  Enhanced real: {enhanced_real.shape}")
    print(f"  Enhanced imag: {enhanced_imag.shape}")
    
    print("\n✓ Model inference successful!\n")


def example_2_loss_functions():
    """
    Example 2: Demonstrate loss functions.
    Shows how different loss functions work.
    """
    print("=" * 60)
    print("Example 2: Loss Functions")
    print("=" * 60)
    
    # Create loss function
    loss_fn = CombinedLoss(
        alpha=0.3,
        lambda_cv=0.5,
        lambda_mag=0.5,
    )
    
    # Create dummy predictions and targets
    batch_size = 2
    freq_bins = 257
    time_frames = 100
    
    enhanced_real = torch.randn(batch_size, 1, freq_bins, time_frames)
    enhanced_imag = torch.randn(batch_size, 1, freq_bins, time_frames)
    clean_real = torch.randn(batch_size, 1, freq_bins, time_frames)
    clean_imag = torch.randn(batch_size, 1, freq_bins, time_frames)
    
    # Compute loss
    total_loss, loss_dict = loss_fn(
        enhanced_real, enhanced_imag,
        clean_real, clean_imag,
    )
    
    print(f"Loss values:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n✓ Loss computation successful!\n")


def example_3_data_processing():
    """
    Example 3: Data processing pipeline.
    Shows STFT/ISTFT operations.
    """
    print("=" * 60)
    print("Example 3: Data Processing (STFT/ISTFT)")
    print("=" * 60)
    
    from speech_enhancement.data_processing import STFTProcessor
    import soundfile as sf
    
    # Create STFT processor
    stft_processor = STFTProcessor(
        n_fft=512,
        hop_length=256,
    )
    
    # Create or load audio
    audio, sr = create_demo_audio(duration=1.0)
    audio = torch.from_numpy(audio).float()
    
    print(f"Input audio shape: {audio.shape}")
    print(f"Sample rate: {sr} Hz\n")
    
    # Compute STFT
    real_part, imag_part = stft_processor.stft(audio)
    print(f"STFT output shapes:")
    print(f"  Real part: {real_part.shape}")
    print(f"  Imag part: {imag_part.shape}")
    
    # Compute magnitude
    magnitude = stft_processor.get_magnitude(real_part, imag_part)
    print(f"  Magnitude: {magnitude.shape}")
    
    # Compute phase
    phase = stft_processor.get_phase(real_part, imag_part)
    print(f"  Phase: {phase.shape}")
    
    # Apply some processing (dummy example: amplify magnitude)
    enhanced_magnitude = magnitude * 1.2
    
    # Reconstruct using original phase
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    enhanced_real = enhanced_magnitude * cos_phase
    enhanced_imag = enhanced_magnitude * sin_phase
    
    # Inverse STFT
    enhanced_audio = stft_processor.istft(enhanced_real, enhanced_imag)
    
    print(f"\nEnhanced audio shape: {enhanced_audio.shape}")
    print(f"✓ STFT/ISTFT processing successful!\n")


def example_4_dataset():
    """
    Example 4: Dataset creation and loading.
    Shows how to create and use datasets.
    """
    print("=" * 60)
    print("Example 4: Dataset Creation and Loading")
    print("=" * 60)
    
    # Create demo data directory
    demo_dir = Path("./data/demo_clean")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample audio files
    import soundfile as sf
    
    print("Creating sample audio files...")
    for i in range(3):
        audio, sr = create_demo_audio(duration=1.0)
        sf.write(f"{demo_dir}/sample_{i}.wav", audio, sr)
        print(f"  Created sample_{i}.wav")
    
    print(f"\nCreating dataset...")
    dataset = SyntheticNoiseDataset(
        clean_dir=str(demo_dir),
        snr_db=10.0,
        num_samples=6,
        max_length=32000,
        seed=42,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create data loader
    collate_fn = FramePaddingCollate()
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Get a batch
    print("\nLoading a batch...")
    noisy_real, noisy_imag, clean_real, clean_imag = next(iter(loader))
    
    print(f"Batch shapes:")
    print(f"  Noisy real: {noisy_real.shape}")
    print(f"  Noisy imag: {noisy_imag.shape}")
    print(f"  Clean real: {clean_real.shape}")
    print(f"  Clean imag: {clean_imag.shape}")
    
    print(f"\n✓ Dataset creation and loading successful!\n")


def example_5_mini_training():
    """
    Example 5: Mini training loop.
    Shows basic training on synthetic data.
    """
    print("=" * 60)
    print("Example 5: Mini Training Loop")
    print("=" * 60)
    
    import soundfile as sf
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create demo data
    demo_dir = Path("./data/demo_clean")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample data...")
    for i in range(2):
        audio, sr = create_demo_audio(duration=1.0)
        sf.write(f"{demo_dir}/sample_{i}.wav", audio, sr)
    
    # Create datasets
    train_dataset = SyntheticNoiseDataset(
        clean_dir=str(demo_dir),
        snr_db=10.0,
        num_samples=4,
        max_length=16000,
    )
    
    val_dataset = SyntheticNoiseDataset(
        clean_dir=str(demo_dir),
        snr_db=10.0,
        num_samples=2,
        max_length=16000,
        seed=123,
    )
    
    # Create loaders
    collate_fn = FramePaddingCollate()
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)
    
    # Create model
    model = SpeechEnhancementNetwork(base_channels=16)  # Smaller for demo
    loss_fn = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir="./demo_checkpoints",
        log_dir="./demo_logs",
    )
    
    # Train for 2 epochs
    print("Starting training...\n")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
    )
    
    print("\n✓ Training loop completed successfully!\n")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Speech Enhancement - Quick Start Examples" + " " * 7 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        # Run examples
        example_1_basic_inference()
        example_2_loss_functions()
        example_3_data_processing()
        example_4_dataset()
        example_5_mini_training()
        
        print("=" * 60)
        print("All examples completed successfully! ✓")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Prepare your own audio data in ./data/clean and ./data/noisy")
        print("2. Train the model: python train.py")
        print("3. Enhance audio: python inference.py --model checkpoints/best_model.pt --input-file input.wav")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

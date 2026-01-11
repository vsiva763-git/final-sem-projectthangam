"""
FIXED Kaggle training script with NaN prevention and validation
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path

from speech_enhancement.model import SpeechEnhancementNetwork
from speech_enhancement.losses import CombinedLoss
from speech_enhancement.trainer import Trainer
from speech_enhancement.dataset import SpeechEnhancementDataset, FramePaddingCollate


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_model_weights(model):
    """Initialize model with small weights to prevent NaN"""
    print("Initializing model with small weights...")
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                torch.nn.init.uniform_(param, -0.1, 0.1)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    print("✓ Model initialized")


def validate_model_output(model, sample_input):
    """Test model with sample input to check for NaN"""
    print("\nValidating model output...")
    model.eval()
    real, imag = sample_input
    real = real[:1].cuda() if torch.cuda.is_available() else real[:1]
    imag = imag[:1].cuda() if torch.cuda.is_available() else imag[:1]
    
    with torch.no_grad():
        out_real, out_imag = model(real, imag)
        
    has_nan = torch.isnan(out_real).any() or torch.isnan(out_imag).any()
    has_inf = torch.isinf(out_real).any() or torch.isinf(out_imag).any()
    all_zero = (out_real.abs().max() < 1e-6) and (out_imag.abs().max() < 1e-6)
    
    print(f"  Output real: min={out_real.min():.4f}, max={out_real.max():.4f}")
    print(f"  Output imag: min={out_imag.min():.4f}, max={out_imag.max():.4f}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  All zeros: {all_zero}")
    
    if has_nan or has_inf or all_zero:
        print("  ⚠️ WARNING: Model output is invalid!")
        return False
    
    print("  ✓ Model output is valid")
    return True


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Get dataset paths
    clean_dir = args.clean_dir or config['data'].get('clean_dir')
    noisy_dir = args.noisy_dir or config['data'].get('noisy_dir')
    
    # Check if running on Kaggle
    kaggle_dataset_path = '/kaggle/input/voicebank-cleantest-esc-crybaby-dog'
    if Path(kaggle_dataset_path).exists():
        clean_dir = f'{kaggle_dataset_path}/clean_testset_wav/clean_testset_wav'
        noisy_dir = f'{kaggle_dataset_path}/noisy_dataset_wav/noisy_dataset_wav'
        print(f"[Kaggle Environment Detected] Using Kaggle dataset paths")
    
    if not clean_dir or not noisy_dir:
        raise ValueError("Must specify clean_dir and noisy_dir")
    
    print("\nConfiguration:")
    print(yaml.dump(config))
    print(f"\nClean audio directory: {clean_dir}")
    print(f"Noisy audio directory: {noisy_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    full_dataset = SpeechEnhancementDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        max_length=config['data'].get('max_length'),
    )
    
    print(f"Total samples in dataset: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("ERROR: Dataset is empty!")
        return
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create data loaders
    collate_fn = FramePaddingCollate()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model
    print("\nCreating model...")
    model = SpeechEnhancementNetwork(base_channels=config['model']['base_channels'])
    
    # Initialize weights properly
    initialize_model_weights(model)
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Validate model before training
    sample_batch = next(iter(train_loader))
    if not validate_model_output(model, (sample_batch[0], sample_batch[1])):
        print("\n❌ Model validation failed! Check model architecture.")
        return
    
    # Loss function
    loss_fn = CombinedLoss(
        alpha=config['loss']['alpha'],
        lambda_cv=config['loss']['lambda_cv'],
        lambda_mag=config['loss']['lambda_mag'],
    )
    
    # Optimizer with smaller learning rate
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        eps=1e-7,  # Smaller epsilon for stability
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir or "./kaggle_checkpoints",
        log_dir=args.log_dir or "./kaggle_logs",
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 80)
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        use_amp=False,  # Disable AMP to prevent NaN
        scheduler=scheduler,
    )
    
    print("\nTraining completed!")
    print("=" * 80)
    print(f"Checkpoints saved to: {args.checkpoint_dir or './kaggle_checkpoints'}")
    print(f"Logs saved to: {args.log_dir or './kaggle_logs'}")
    print("\n✓ Best model: ./kaggle_checkpoints/best_model.pt")
    print("=" * 80)
    
    # Final validation
    print("\nFinal model validation...")
    if validate_model_output(model, (sample_batch[0], sample_batch[1])):
        print("✓ Model is ready for inference!")
    else:
        print("⚠️ Warning: Final model may have issues")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speech enhancement model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--clean-dir", type=str, default=None)
    parser.add_argument("--noisy-dir", type=str, default=None)
    
    args = parser.parse_args()
    main(args)

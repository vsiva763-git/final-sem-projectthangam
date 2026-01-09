"""
Main training script for speech enhancement model.
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
from speech_enhancement.dataset import SyntheticNoiseDataset, FramePaddingCollate


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function."""
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'model': {
                'base_channels': 32,
            },
            'training': {
                'epochs': 50,
                'batch_size': 4,
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
            },
            'loss': {
                'alpha': 0.3,
                'lambda_cv': 0.5,
                'lambda_mag': 0.5,
            },
            'data': {
                'n_fft': 512,
                'hop_length': 256,
                'max_length': 32000,
            },
        }
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    print("Configuration:")
    print(yaml.dump(config))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets (using synthetic data for demo)
    print("\nCreating datasets...")
    
    # For demo purposes, we'll create synthetic datasets
    # In practice, you would use real audio files
    train_dataset = SyntheticNoiseDataset(
        clean_dir=args.clean_dir,
        snr_db=10.0,
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        num_samples=args.train_samples,
        max_length=config['data'].get('max_length'),
    )
    
    val_dataset = SyntheticNoiseDataset(
        clean_dir=args.clean_dir,
        snr_db=10.0,
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        num_samples=args.val_samples,
        max_length=config['data'].get('max_length'),
        seed=123,  # Different seed for validation
    )
    
    # Create data loaders
    collate_fn = FramePaddingCollate()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = SpeechEnhancementNetwork(
        base_channels=config['model']['base_channels'],
    )
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = CombinedLoss(
        alpha=config['loss']['alpha'],
        lambda_cv=config['loss']['lambda_cv'],
        lambda_mag=config['loss']['lambda_mag'],
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0),
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    # Train model
    print("\nStarting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        use_amp=args.use_amp,
    )
    
    print("\nTraining completed!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speech enhancement model")
    
    # Data arguments
    parser.add_argument(
        "--clean-dir",
        type=str,
        default="./data/clean",
        help="Directory containing clean audio files",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=100,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="Number of validation samples to generate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    
    # Directory arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory to save logs",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    
    args = parser.parse_args()
    
    # Create clean data directory for demo (will be populated with sample files)
    Path(args.clean_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)

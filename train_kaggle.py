"""
Train speech enhancement model using Kaggle VoiceBank dataset.
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
    
    # Get dataset paths from config or arguments
    clean_dir = args.clean_dir or config['data'].get('clean_dir')
    noisy_dir = args.noisy_dir or config['data'].get('noisy_dir')
    
    if not clean_dir or not noisy_dir:
        raise ValueError("Must specify clean_dir and noisy_dir in config or command line")
    
    print("Configuration:")
    print(yaml.dump(config))
    print(f"\nClean audio directory: {clean_dir}")
    print(f"Noisy audio directory: {noisy_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    
    # Full dataset
    full_dataset = SpeechEnhancementDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        max_length=config['data'].get('max_length'),
    )
    
    print(f"Total samples in dataset: {len(full_dataset)}")
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    collate_fn = FramePaddingCollate()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = SpeechEnhancementNetwork(
        base_channels=config['model']['base_channels'],
    )
    
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
    print("=" * 80)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        use_amp=args.use_amp,
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train speech enhancement model on Kaggle VoiceBank dataset"
    )
    
    # Data arguments
    parser.add_argument(
        "--clean-dir",
        type=str,
        help="Directory containing clean audio files (overrides config)",
    )
    parser.add_argument(
        "--noisy-dir",
        type=str,
        help="Directory containing noisy audio files (overrides config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for data loading",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for optimizer (overrides config)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision for faster training",
    )
    
    # Directory arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./kaggle_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./kaggle_logs",
        help="Directory to save logs",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default_config.yaml",
        help="Path to configuration YAML file",
    )
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)

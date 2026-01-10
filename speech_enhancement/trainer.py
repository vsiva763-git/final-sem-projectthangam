"""
Training utilities for speech enhancement model.
Includes trainer class, checkpoint management, and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm


class ModelCheckpoint:
    """
    Save and manage model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best: bool = True,
        best_metric: str = "loss",
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model based on metric
            best_metric: Metric to track for best model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_value = float('inf')
    
    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metric_value: float,
        is_best: bool = False,
    ) -> None:
        """
        Save checkpoint.
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state
            metric_value: Current metric value
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_value': metric_value,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {self.best_metric}={metric_value:.4f}")
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Path to checkpoint (uses best model if None)
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


class TrainingMonitor:
    """
    Monitor training metrics and log progress.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train': {},
            'val': {},
        }
        
        self.log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def update(
        self,
        stage: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Update metrics for a stage.
        
        Args:
            stage: 'train' or 'val'
            epoch: Current epoch
            metrics: Dictionary of metric values
        """
        if epoch not in self.history[stage]:
            self.history[stage][epoch] = {}
        
        self.history[stage][epoch].update(metrics)
    
    def save(self) -> None:
        """
        Save training history to file.
        """
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def print_metrics(
        self,
        stage: str,
        epoch: int,
    ) -> None:
        """
        Print metrics for a stage and epoch.
        
        Args:
            stage: 'train' or 'val'
            epoch: Epoch to print
        """
        if epoch in self.history[stage]:
            metrics = self.history[stage][epoch]
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"[{stage.upper()}] Epoch {epoch}: {metric_str}")


class Trainer:
    """
    Main trainer class for speech enhancement model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Speech enhancement model
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to use (cuda/cpu)
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir)
        self.monitor = TrainingMonitor(log_dir)
        
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            use_amp: Use automatic mixed precision
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            noisy_real, noisy_imag, clean_real, clean_imag = [
                b.to(self.device) for b in batch
            ]
            
            self.optimizer.zero_grad()
            
            if use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    enhanced_real, enhanced_imag = self.model(noisy_real, noisy_imag)
                    loss, loss_dict = self.loss_fn(
                        enhanced_real, enhanced_imag,
                        clean_real, clean_imag,
                    )
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward-backward
                enhanced_real, enhanced_imag = self.model(noisy_real, noisy_imag)
                loss, loss_dict = self.loss_fn(
                    enhanced_real, enhanced_imag,
                    clean_real, clean_imag,
                )
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.optimizer.step()
            
            # Skip NaN losses
            loss_value = loss.item()
            if not torch.isnan(torch.tensor(loss_value)):
                total_loss += loss_value
                num_batches += 1
            
            # Update progress bar with safe division
            avg_loss = total_loss / max(num_batches, 1)
            pbar.set_postfix({
                'loss': avg_loss,
                'loss_cv': loss_dict.get('loss_cv', 0.0),
                'loss_mag': loss_dict.get('loss_mag', 0.0),
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            
            for batch in pbar:
                # Unpack batch
                noisy_real, noisy_imag, clean_real, clean_imag = [
                    b.to(self.device) for b in batch
                ]
                
                # Forward pass
                enhanced_real, enhanced_imag = self.model(noisy_real, noisy_imag)
                loss, loss_dict = self.loss_fn(
                    enhanced_real, enhanced_imag,
                    clean_real, clean_imag,
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        use_amp: bool = False,
    ) -> None:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            use_amp: Use automatic mixed precision
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, use_amp=use_amp)
            self.monitor.update('train', epoch, train_metrics)
            self.monitor.print_metrics('train', epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.monitor.update('val', epoch, val_metrics)
            self.monitor.print_metrics('val', epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            self.checkpoint_manager.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metric_value=val_metrics['loss'],
                is_best=is_best,
            )
        
        # Save final training history
        self.monitor.save()
        print("\nTraining completed!")

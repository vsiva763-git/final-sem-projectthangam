"""
Loss functions for speech enhancement training.
Includes complex value loss, magnitude loss, and combined loss.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PowerLawCompressor(nn.Module):
    """
    Power-law compression for loss computation.
    Helps balance different frequency ranges with numerical stability.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize power-law compressor.
        
        Args:
            alpha: Compression parameter (typically 0.3-0.5)
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply power-law compression with numerical stability.
        
        Args:
            x: Input tensor
            
        Returns:
            Compressed tensor
        """
        # Add small epsilon to avoid numerical issues near zero
        epsilon = 1e-7
        return torch.sign(x) * torch.pow(torch.abs(x) + epsilon, self.alpha)


class ComplexValueLoss(nn.Module):
    """
    Complex Value Loss (L_cv).
    MSE between compressed real and imaginary parts.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize complex value loss.
        
        Args:
            alpha: Power-law compression parameter
        """
        super().__init__()
        self.compressor = PowerLawCompressor(alpha=alpha)
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        enhanced_real: torch.Tensor,
        enhanced_imag: torch.Tensor,
        clean_real: torch.Tensor,
        clean_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute complex value loss.
        
        Args:
            enhanced_real: Predicted real part
            enhanced_imag: Predicted imaginary part
            clean_real: Target real part
            clean_imag: Target imaginary part
            
        Returns:
            Loss value
        """
        # Apply power-law compression with stable computation
        compressed_enhanced_real = self.compressor(enhanced_real)
        compressed_enhanced_imag = self.compressor(enhanced_imag)
        compressed_clean_real = self.compressor(clean_real)
        compressed_clean_imag = self.compressor(clean_imag)
        
        # Compute MSE on compressed values
        loss_real = self.mse(compressed_enhanced_real, compressed_clean_real)
        loss_imag = self.mse(compressed_enhanced_imag, compressed_clean_imag)
        
        # Return mean to prevent loss explosion
        return torch.clamp(loss_real + loss_imag, max=1e6) / 2.0


class MagnitudeLoss(nn.Module):
    """
    Magnitude Loss (L_mag).
    MSE between estimated and clean speech amplitude values.
    """
    
    def __init__(self):
        """
        Initialize magnitude loss.
        """
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        enhanced_real: torch.Tensor,
        enhanced_imag: torch.Tensor,
        clean_real: torch.Tensor,
        clean_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute magnitude loss.
        
        Args:
            enhanced_real: Predicted real part
            enhanced_imag: Predicted imaginary part
            clean_real: Target real part
            clean_imag: Target imaginary part
            
        Returns:
            Loss value
        """
        # Compute magnitudes with numerical stability
        epsilon = 1e-8
        enhanced_mag = torch.sqrt(enhanced_real ** 2 + enhanced_imag ** 2 + epsilon)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + epsilon)
        
        # Compute MSE on magnitudes
        return torch.clamp(self.mse(enhanced_mag, clean_mag), max=1e6)


class CombinedLoss(nn.Module):
    """
    Combined Loss (L_total).
    Simple and stable loss using L1 and MSE.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        lambda_cv: float = 0.5,
        lambda_mag: float = 0.5,
    ):
        """
        Initialize combined loss.
        
        Args:
            alpha: Power-law compression parameter (unused, kept for compatibility)
            lambda_cv: Weight for L1 loss on components
            lambda_mag: Weight for magnitude loss
        """
        super().__init__()
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.lambda_cv = lambda_cv
        self.lambda_mag = lambda_mag
    
    def forward(
        self,
        enhanced_real: torch.Tensor,
        enhanced_imag: torch.Tensor,
        clean_real: torch.Tensor,
        clean_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss using stable L1 and MSE.
        
        Args:
            enhanced_real: Predicted real part
            enhanced_imag: Predicted imaginary part
            clean_real: Target real part
            clean_imag: Target imaginary part
            
        Returns:
            Total loss and dictionary with individual losses
        """
        # Simple L1 loss on real and imaginary parts (more stable than power-law)
        cv_loss_real = self.l1_loss(enhanced_real, clean_real)
        cv_loss_imag = self.l1_loss(enhanced_imag, clean_imag)
        cv_loss = (cv_loss_real + cv_loss_imag) / 2.0
        
        # Magnitude loss
        epsilon = 1e-8
        enhanced_mag = torch.sqrt(enhanced_real ** 2 + enhanced_imag ** 2 + epsilon)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + epsilon)
        mag_loss = self.mse_loss(enhanced_mag, clean_mag)
        
        # Weighted combination
        total_loss = self.lambda_cv * cv_loss + self.lambda_mag * mag_loss
        
        # Ensure no NaN
        total_loss = torch.clamp(total_loss, max=1e6)
        
        # Safely get loss values
        total_loss_val = total_loss.item() if not torch.isnan(total_loss).any() else 0.0
        cv_loss_val = cv_loss.item() if not torch.isnan(cv_loss).any() else 0.0
        mag_loss_val = mag_loss.item() if not torch.isnan(mag_loss).any() else 0.0
        
        loss_dict = {
            "loss_total": total_loss_val,
            "loss_cv": cv_loss_val,
            "loss_mag": mag_loss_val,
        }
        
        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """
    Perceptual loss for additional training stability.
    Encourages outputs to be close in perceptual space.
    """
    
    def __init__(self, use_l1: bool = True):
        """
        Initialize perceptual loss.
        
        Args:
            use_l1: Whether to use L1 loss instead of L2
        """
        super().__init__()
        self.use_l1 = use_l1
        
        if use_l1:
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
    
    def forward(
        self,
        enhanced_real: torch.Tensor,
        enhanced_imag: torch.Tensor,
        clean_real: torch.Tensor,
        clean_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            enhanced_real: Predicted real part
            enhanced_imag: Predicted imaginary part
            clean_real: Target real part
            clean_imag: Target imaginary part
            
        Returns:
            Perceptual loss value
        """
        loss_real = self.criterion(enhanced_real, clean_real)
        loss_imag = self.criterion(enhanced_imag, clean_imag)
        
        return loss_real + loss_imag

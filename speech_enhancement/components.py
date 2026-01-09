"""
Core architecture components for speech enhancement network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class UnifiedDiscoveryUnit(nn.Module):
    """
    Unified Discovery Unit (UDU) with dual expert segments and gating.
    Replaces standard convolutional layers for cooperative learning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_1: int = 3,
        kernel_size_2: int = 5,
        stride: int = 1,
        padding: int = 1,
    ):
        """
        Initialize UDU with two expert segments.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size_1: Kernel size for first expert (not used, for compatibility)
            kernel_size_2: Kernel size for second expert (not used, for compatibility)
            stride: Convolution stride
            padding: Convolution padding
        """
        super().__init__()
        
        # Use a standard kernel size for both experts
        kernel_size = 3
        
        # Expert segment 1
        self.expert_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Expert segment 2 (with different internal structure)
        self.expert_2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Gating mechanism to balance both experts
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gating mechanism.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W)
            
        Returns:
            Output tensor of shape (B, C_out, H', W') with gated expert combination
        """
        # Expert outputs - both should have same shape now
        expert_1_out = self.expert_1(x)
        expert_2_out = self.expert_2(x)
        
        # Gating weights
        gate_weight = self.gate(x)
        
        # Interpolate gate to match expert output size (in case of stride)
        if gate_weight.shape != expert_1_out.shape:
            gate_weight = torch.nn.functional.interpolate(
                gate_weight,
                size=expert_1_out.shape[-2:],
                mode='nearest',
            )
        
        # Gate-weighted combination
        output = expert_1_out * gate_weight + expert_2_out * (1 - gate_weight)
        
        return output


class SpatialChannelAttention(nn.Module):
    """
    Spatial-Channel Attention module for TSAIC.
    Uses teachable weight coefficients for combining features.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize spatial-channel attention.
        
        Args:
            channels: Number of channels
            reduction: Reduction factor for bottleneck
        """
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
        # Teachable weight coefficients
        self.channel_weight = nn.Parameter(torch.ones(1))
        self.spatial_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial-channel attention.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            
        Returns:
            Attention-weighted output
        """
        # Channel attention
        batch, channels, freq, time = x.size()
        
        avg_out = self.channel_fc(self.avg_pool(x).view(batch, channels))
        max_out = self.channel_fc(self.max_pool(x).view(batch, channels))
        channel_out = (avg_out + max_out).sigmoid().view(batch, channels, 1, 1)
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial_in = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_out = self.spatial_conv(spatial_in).sigmoid()
        
        # Combine with teachable weights
        channel_attention = channel_out * self.channel_weight
        spatial_attention = spatial_out * self.spatial_weight
        
        return x * channel_attention * spatial_attention


class TemporalAttentionTransformer(nn.Module):
    """
    Temporal Attention Transformer (TAT) for subband modeling.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        """
        Initialize temporal attention transformer.
        
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal attention.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            
        Returns:
            Output tensor with temporal attention applied
        """
        batch, channels, freq, time = x.size()
        
        # Reshape for temporal attention: treat freq as batch dimension
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, freq, channels, time]
        x = x.view(batch * freq, channels, time).transpose(1, 2)  # [batch*freq, time, channels]
        
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Reshape back
        x = x.transpose(1, 2).view(batch, freq, channels, time)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        return x


class SpectralAttentionTransformer(nn.Module):
    """
    Spectral Attention Transformer (SAT) for full-band dependencies.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        """
        Initialize spectral attention transformer.
        
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spectral attention.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            
        Returns:
            Output tensor with spectral attention applied
        """
        batch, channels, freq, time = x.size()
        
        # Reshape for spectral attention: treat time as batch dimension
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, time, channels, freq]
        x = x.view(batch * time, channels, freq).transpose(1, 2)  # [batch*time, freq, channels]
        
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Reshape back
        x = x.transpose(1, 2).view(batch, time, channels, freq)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x


class TemporalSpectralCrossAttention(nn.Module):
    """
    Temporal-Spectral Cross-Attention Component (TSCAC).
    Integrates TAT for subband modeling and SAT for full-band dependencies.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        """
        Initialize TSCAC.
        
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.tat = TemporalAttentionTransformer(channels, num_heads)
        self.sat = SpectralAttentionTransformer(channels, num_heads)
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining temporal and spectral attention.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            
        Returns:
            Output tensor with cross-attention applied
        """
        # Apply temporal and spectral attention
        tat_out = self.tat(x)
        sat_out = self.sat(x)
        
        # Fuse the two attention outputs
        fused = torch.cat([tat_out, sat_out], dim=1)
        output = self.fusion(fused)
        
        return output


class GaussianWeightedProgressiveStructure(nn.Module):
    """
    Gaussian-Weighted Progressive Structure (GWPS).
    Integrates information flow using Gaussian weight coefficients.
    """
    
    def __init__(self, channels: int, depth: int = 4):
        """
        Initialize GWPS.
        
        Args:
            channels: Number of channels
            depth: Number of layers
        """
        super().__init__()
        self.depth = depth
        
        # Gaussian weight coefficients for each layer
        self.gaussian_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([self._gaussian(i, depth)]))
            for i in range(depth)
        ])
    
    @staticmethod
    def _gaussian(layer: int, total_layers: int, sigma: float = 1.0) -> float:
        """
        Compute Gaussian weight for given layer.
        
        Args:
            layer: Current layer index
            total_layers: Total number of layers
            sigma: Standard deviation of Gaussian
            
        Returns:
            Gaussian weight
        """
        mean = total_layers / 2
        exponent = -((layer - mean) ** 2) / (2 * sigma ** 2)
        return float(torch.exp(torch.tensor(exponent)))
    
    def forward(
        self,
        layer_outputs: list,
    ) -> torch.Tensor:
        """
        Progressively combine layer outputs with Gaussian weights.
        
        Args:
            layer_outputs: List of outputs from each layer
            
        Returns:
            Weighted combination of layer outputs
        """
        weighted_sum = None
        total_weight = 0
        
        for i, output in enumerate(layer_outputs):
            weight = self.gaussian_weights[i]
            
            if weighted_sum is None:
                weighted_sum = output * weight
            else:
                weighted_sum = weighted_sum + output * weight
            
            total_weight += weight
        
        # Normalize by total weight
        return weighted_sum / (total_weight + 1e-8)


class TwinSegmentAttentionIntegration(nn.Module):
    """
    Twin-Segment Attention Integration Component (TSAIC).
    Combines spatial and channel features using teachable weights.
    """
    
    def __init__(self, channels: int):
        """
        Initialize TSAIC.
        
        Args:
            channels: Number of channels
        """
        super().__init__()
        
        self.attention = SpatialChannelAttention(channels)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial-channel attention and projection.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            
        Returns:
            Integrated output
        """
        attention_out = self.attention(x)
        output = self.projection(attention_out)
        return output

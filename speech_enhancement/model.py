"""
Main speech enhancement model architecture.
Encoder-Decoder structure with complex spectral mapping.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .components import (
    UnifiedDiscoveryUnit,
    TemporalSpectralCrossAttention,
    TwinSegmentAttentionIntegration,
    GaussianWeightedProgressiveStructure,
)


class Encoder(nn.Module):
    """
    Encoder with 4 UDU units for dimension compression and feature extraction.
    """
    
    def __init__(self, in_channels: int = 2, base_channels: int = 32):
        """
        Initialize encoder.
        
        Args:
            in_channels: Number of input channels (2 for real + imaginary)
            base_channels: Base number of channels
        """
        super().__init__()
        
        self.udu_1 = UnifiedDiscoveryUnit(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_2 = UnifiedDiscoveryUnit(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_3 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_4 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through encoder with skip connections.
        
        Args:
            x: Input tensor [batch, 1, freq, time]
            
        Returns:
            Encoder output and skip connection features
        """
        skip_1 = self.udu_1(x)
        skip_2 = self.udu_2(skip_1)
        skip_3 = self.udu_3(skip_2)
        skip_4 = self.udu_4(skip_3)
        
        return skip_4, [skip_3, skip_2, skip_1]


class IntermediateLayer(nn.Module):
    """
    Intermediate layer combining 2D convolution and GWPS.
    """
    
    def __init__(self, channels: int = 256):
        """
        Initialize intermediate layer.
        
        Args:
            channels: Number of input channels
        """
        super().__init__()
        
        # 2D convolution to halve channel numbers
        self.conv_2d = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # TSCAC for attention
        self.tscac = TemporalSpectralCrossAttention(
            channels=channels // 2,
            num_heads=4,
        )
        
        # GWPS for information integration
        self.gwps = GaussianWeightedProgressiveStructure(
            channels=channels // 2,
            depth=4,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through intermediate layer.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            
        Returns:
            Intermediate output with TSCAC and GWPS applied
        """
        # Apply 2D convolution
        conv_out = self.conv_2d(x)
        
        # Apply TSCAC
        attention_out = self.tscac(conv_out)
        
        return attention_out


class RealDecoder(nn.Module):
    """
    Decoder for real part with 4 UDU units (RUDU).
    """
    
    def __init__(self, base_channels: int = 32):
        """
        Initialize real decoder.
        
        Args:
            base_channels: Base number of channels
        """
        super().__init__()
        
        self.udu_1 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_2 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 4 + base_channels * 4,
            out_channels=base_channels * 2,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_3 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 2 + base_channels * 2,
            out_channels=base_channels,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_4 = UnifiedDiscoveryUnit(
            in_channels=base_channels + base_channels,
            out_channels=1,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: list,
    ) -> torch.Tensor:
        """
        Forward pass through real decoder with skip connections.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            skip_connections: List of skip connection tensors
            
        Returns:
            Decoded real part [batch, 1, freq, time]
        """
        out = self.udu_1(x)
        out = torch.cat([out, skip_connections[0]], dim=1)
        
        out = self.udu_2(out)
        out = torch.cat([out, skip_connections[1]], dim=1)
        
        out = self.udu_3(out)
        out = torch.cat([out, skip_connections[2]], dim=1)
        
        out = self.udu_4(out)
        
        return out


class ImaginaryDecoder(nn.Module):
    """
    Decoder for imaginary part with 4 UDU units (IUDU).
    """
    
    def __init__(self, base_channels: int = 32):
        """
        Initialize imaginary decoder.
        
        Args:
            base_channels: Base number of channels
        """
        super().__init__()
        
        self.udu_1 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 4,
            out_channels=base_channels * 4,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_2 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 4 + base_channels * 4,
            out_channels=base_channels * 2,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_3 = UnifiedDiscoveryUnit(
            in_channels=base_channels * 2 + base_channels * 2,
            out_channels=base_channels,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
        
        self.udu_4 = UnifiedDiscoveryUnit(
            in_channels=base_channels + base_channels,
            out_channels=1,
            kernel_size_1=3,
            kernel_size_2=5,
            stride=1,
            padding=1,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: list,
    ) -> torch.Tensor:
        """
        Forward pass through imaginary decoder with skip connections.
        
        Args:
            x: Input tensor [batch, channels, freq, time]
            skip_connections: List of skip connection tensors
            
        Returns:
            Decoded imaginary part [batch, 1, freq, time]
        """
        out = self.udu_1(x)
        out = torch.cat([out, skip_connections[0]], dim=1)
        
        out = self.udu_2(out)
        out = torch.cat([out, skip_connections[1]], dim=1)
        
        out = self.udu_3(out)
        out = torch.cat([out, skip_connections[2]], dim=1)
        
        out = self.udu_4(out)
        
        return out


class SpeechEnhancementNetwork(nn.Module):
    """
    Complete speech enhancement network with encoder-decoder architecture.
    Performs complex spectral mapping for mono-channel speech enhancement.
    """
    
    def __init__(self, base_channels: int = 32):
        """
        Initialize speech enhancement network.
        
        Args:
            base_channels: Base number of channels throughout the network
        """
        super().__init__()
        
        self.encoder = Encoder(in_channels=2, base_channels=base_channels)
        self.intermediate = IntermediateLayer(channels=base_channels * 8)
        self.real_decoder = RealDecoder(base_channels=base_channels)
        self.imag_decoder = ImaginaryDecoder(base_channels=base_channels)
    
    def forward(
        self,
        real_part: torch.Tensor,
        imag_part: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through speech enhancement network.
        
        Args:
            real_part: Real part of complex spectrum [batch, 1, freq, time]
            imag_part: Imaginary part of complex spectrum [batch, 1, freq, time]
            
        Returns:
            Enhanced real and imaginary parts [batch, 1, freq, time]
        """
        # Concatenate real and imaginary parts
        x = torch.cat([real_part, imag_part], dim=1)  # [batch, 2, freq, time]
        
        # Encoder
        encoder_out, skip_connections = self.encoder(x)
        
        # Intermediate layer
        intermediate_out = self.intermediate(encoder_out)
        
        # Decoders for real and imaginary parts
        enhanced_real = self.real_decoder(intermediate_out, skip_connections)
        enhanced_imag = self.imag_decoder(intermediate_out, skip_connections)
        
        return enhanced_real, enhanced_imag

"""
Data processing utilities for STFT/ISTFT and audio handling.
"""

import torch
import numpy as np
import scipy.signal as signal
from typing import Tuple, Union


class STFTProcessor:
    """
    Handles STFT and ISTFT transformations for audio processing.
    """
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        window: str = "hann",
        center: bool = True,
    ):
        """
        Initialize STFT processor.
        
        Args:
            n_fft: FFT size
            hop_length: Number of samples between successive frames
            window: Window type (hann, hamming, etc.)
            center: Whether to center the signal
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.n_freq_bins = n_fft // 2 + 1
        
    def stft(self, audio: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute STFT of audio signal.
        
        Args:
            audio: Input audio waveform [time_samples]
            
        Returns:
            Complex spectrum as real and imaginary parts [freq_bins, time_frames]
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        # Compute STFT using scipy
        frequencies, times, stft_matrix = signal.stft(
            audio,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window=self.window,
        )
        
        # Extract real and imaginary parts
        real_part = torch.from_numpy(np.real(stft_matrix)).float()
        imag_part = torch.from_numpy(np.imag(stft_matrix)).float()
        
        return real_part, imag_part
    
    def istft(
        self,
        real_part: torch.Tensor,
        imag_part: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inverse STFT to recover audio signal.
        
        Args:
            real_part: Real part of STFT [freq_bins, time_frames]
            imag_part: Imaginary part of STFT [freq_bins, time_frames]
            
        Returns:
            Reconstructed audio waveform [time_samples]
        """
        if isinstance(real_part, torch.Tensor):
            real_part = real_part.cpu().numpy()
        if isinstance(imag_part, torch.Tensor):
            imag_part = imag_part.cpu().numpy()
        
        # Combine real and imaginary parts
        stft_matrix = real_part + 1j * imag_part
        
        # Compute inverse STFT using scipy
        _, audio = signal.istft(
            stft_matrix,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window=self.window,
        )
        
        return torch.from_numpy(audio).float()
    
    def get_magnitude(
        self,
        real_part: torch.Tensor,
        imag_part: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute magnitude from real and imaginary parts.
        
        Args:
            real_part: Real part [freq_bins, time_frames]
            imag_part: Imaginary part [freq_bins, time_frames]
            
        Returns:
            Magnitude spectrum [freq_bins, time_frames]
        """
        return torch.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8)
    
    def get_phase(
        self,
        real_part: torch.Tensor,
        imag_part: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phase from real and imaginary parts.
        
        Args:
            real_part: Real part [freq_bins, time_frames]
            imag_part: Imaginary part [freq_bins, time_frames]
            
        Returns:
            Phase [freq_bins, time_frames]
        """
        return torch.atan2(imag_part, real_part)


class PowerLawCompression:
    """
    Apply power-law compression to spectral features.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize power-law compression.
        
        Args:
            alpha: Compression parameter (typically 0.3-0.5)
        """
        self.alpha = alpha
    
    def compress(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Apply power-law compression.
        
        Args:
            spectrum: Input spectrum
            
        Returns:
            Compressed spectrum
        """
        return torch.sign(spectrum) * torch.abs(spectrum) ** self.alpha
    
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """
        Reverse power-law compression.
        
        Args:
            compressed: Compressed spectrum
            
        Returns:
            Decompressed spectrum
        """
        return torch.sign(compressed) * torch.abs(compressed) ** (1 / self.alpha)


class AudioNormalizer:
    """
    Normalize audio signal to fixed range.
    """
    
    @staticmethod
    def normalize(audio: Union[np.ndarray, torch.Tensor], target_db: float = -20.0) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Input audio
            target_db: Target dB level
            
        Returns:
            Normalized audio
        """
        is_torch = isinstance(audio, torch.Tensor)
        if is_torch:
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio
        
        # Compute RMS
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms < 1e-8:
            rms = 1e-8
        
        # Convert target dB to linear scale
        target_linear = 10 ** (target_db / 20.0)
        
        # Scale audio
        scaled_audio = audio_np * (target_linear / rms)
        
        if is_torch:
            return torch.from_numpy(scaled_audio).float()
        else:
            return scaled_audio
    
    @staticmethod
    def denormalize(audio: Union[np.ndarray, torch.Tensor], reference_rms: float) -> Union[np.ndarray, torch.Tensor]:
        """
        Restore audio to original RMS level.
        
        Args:
            audio: Normalized audio
            reference_rms: Original RMS value
            
        Returns:
            Audio restored to original level
        """
        return audio * reference_rms

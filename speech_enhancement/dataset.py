"""
Dataset utilities for speech enhancement.
Includes dataset classes and data loading utilities.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional
import soundfile as sf
from pathlib import Path
from .data_processing import STFTProcessor


class SpeechEnhancementDataset(Dataset):
    """
    Dataset for speech enhancement.
    Loads noisy and clean audio pairs.
    """
    
    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        n_fft: int = 512,
        hop_length: int = 256,
        max_length: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            noisy_dir: Directory containing noisy audio files
            clean_dir: Directory containing clean audio files
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            max_length: Maximum audio length in samples (None for no limit)
        """
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.max_length = max_length
        
        self.stft_processor = STFTProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        # Get list of audio files
        self.noisy_files = sorted(self.noisy_dir.glob("*.wav"))
        self.clean_files = sorted(self.clean_dir.glob("*.wav"))
        
        # Ensure matching file counts
        assert len(self.noisy_files) == len(self.clean_files), \
            f"Mismatch in number of files: {len(self.noisy_files)} vs {len(self.clean_files)}"
    
    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.noisy_files)
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (noisy_real, noisy_imag, clean_real, clean_imag)
        """
        # Load audio files
        noisy_audio, _ = sf.read(str(self.noisy_files[idx]))
        clean_audio, _ = sf.read(str(self.clean_files[idx]))
        
        # Limit length if specified
        if self.max_length is not None:
            noisy_audio = noisy_audio[:self.max_length]
            clean_audio = clean_audio[:self.max_length]
        
        # Convert to torch tensors
        noisy_audio = torch.from_numpy(noisy_audio).float()
        clean_audio = torch.from_numpy(clean_audio).float()
        
        # Compute STFT
        noisy_real, noisy_imag = self.stft_processor.stft(noisy_audio)
        clean_real, clean_imag = self.stft_processor.stft(clean_audio)
        
        # Add channel dimension
        noisy_real = noisy_real.unsqueeze(0)  # [1, freq, time]
        noisy_imag = noisy_imag.unsqueeze(0)
        clean_real = clean_real.unsqueeze(0)
        clean_imag = clean_imag.unsqueeze(0)
        
        return noisy_real, noisy_imag, clean_real, clean_imag


class SyntheticNoiseDataset(Dataset):
    """
    Synthetic dataset generator for testing and demonstration.
    Creates noisy versions of clean speech by adding synthetic noise.
    """
    
    def __init__(
        self,
        clean_dir: str,
        snr_db: float = 10.0,
        n_fft: int = 512,
        hop_length: int = 256,
        num_samples: int = 100,
        max_length: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize synthetic noise dataset.
        
        Args:
            clean_dir: Directory containing clean audio files
            snr_db: Signal-to-Noise Ratio in dB
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            num_samples: Total number of samples to generate
            max_length: Maximum audio length in samples
            seed: Random seed for reproducibility
        """
        self.clean_dir = Path(clean_dir)
        self.snr_db = snr_db
        self.max_length = max_length
        self.num_samples = num_samples
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.stft_processor = STFTProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        # Get list of audio files
        self.clean_files = sorted(self.clean_dir.glob("*.wav"))
        assert len(self.clean_files) > 0, f"No audio files found in {clean_dir}"
    
    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return self.num_samples
    
    @staticmethod
    def add_gaussian_noise(
        audio: np.ndarray,
        snr_db: float,
    ) -> np.ndarray:
        """
        Add Gaussian noise to audio at specified SNR.
        
        Args:
            audio: Input audio
            snr_db: Signal-to-Noise Ratio in dB
            
        Returns:
            Noisy audio
        """
        # Compute signal power
        signal_power = np.mean(audio ** 2)
        
        # Compute noise power based on SNR
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate Gaussian noise
        noise = np.random.normal(
            loc=0.0,
            scale=np.sqrt(noise_power),
            size=audio.shape,
        )
        
        return audio + noise
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (noisy_real, noisy_imag, clean_real, clean_imag)
        """
        # Randomly select a clean audio file
        clean_file = self.clean_files[idx % len(self.clean_files)]
        
        # Load clean audio
        clean_audio, _ = sf.read(str(clean_file))
        clean_audio = clean_audio.astype(np.float32)
        
        # Limit length if specified
        if self.max_length is not None:
            clean_audio = clean_audio[:self.max_length]
        
        # Add noise
        noisy_audio = self.add_gaussian_noise(clean_audio, self.snr_db)
        
        # Convert to torch tensors
        clean_audio = torch.from_numpy(clean_audio).float()
        noisy_audio = torch.from_numpy(noisy_audio).float()
        
        # Compute STFT
        noisy_real, noisy_imag = self.stft_processor.stft(noisy_audio)
        clean_real, clean_imag = self.stft_processor.stft(clean_audio)
        
        # Add channel dimension
        noisy_real = noisy_real.unsqueeze(0)
        noisy_imag = noisy_imag.unsqueeze(0)
        clean_real = clean_real.unsqueeze(0)
        clean_imag = clean_imag.unsqueeze(0)
        
        return noisy_real, noisy_imag, clean_real, clean_imag


class FramePaddingCollate:
    """
    Custom collate function to handle variable-length frames.
    Pads sequences to the same length within a batch.
    """
    
    def __init__(self, pad_value: float = 0.0):
        """
        Initialize collate function.
        
        Args:
            pad_value: Value to use for padding
        """
        self.pad_value = pad_value
    
    def __call__(self, batch):
        """
        Collate function.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched tensors
        """
        noisy_real_list = []
        noisy_imag_list = []
        clean_real_list = []
        clean_imag_list = []
        
        # Collect max dimensions
        max_freq = max(item[0].shape[1] for item in batch)
        max_time = max(item[0].shape[2] for item in batch)
        
        for noisy_real, noisy_imag, clean_real, clean_imag in batch:
            # Pad if necessary
            freq_pad = max_freq - noisy_real.shape[1]
            time_pad = max_time - noisy_real.shape[2]
            
            if freq_pad > 0 or time_pad > 0:
                pad_spec = (0, time_pad, 0, freq_pad)
                noisy_real = torch.nn.functional.pad(noisy_real, pad_spec, value=self.pad_value)
                noisy_imag = torch.nn.functional.pad(noisy_imag, pad_spec, value=self.pad_value)
                clean_real = torch.nn.functional.pad(clean_real, pad_spec, value=self.pad_value)
                clean_imag = torch.nn.functional.pad(clean_imag, pad_spec, value=self.pad_value)
            
            noisy_real_list.append(noisy_real)
            noisy_imag_list.append(noisy_imag)
            clean_real_list.append(clean_real)
            clean_imag_list.append(clean_imag)
        
        # Stack into batch
        return (
            torch.stack(noisy_real_list, dim=0),
            torch.stack(noisy_imag_list, dim=0),
            torch.stack(clean_real_list, dim=0),
            torch.stack(clean_imag_list, dim=0),
        )

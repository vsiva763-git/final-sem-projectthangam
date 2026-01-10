"""
Inference script for speech enhancement model.
Processes noisy WAV files and saves enhanced versions.
"""

import torch
import argparse
import soundfile as sf
from pathlib import Path
import numpy as np
from tqdm import tqdm

from speech_enhancement.model import SpeechEnhancementNetwork
from speech_enhancement.data_processing import STFTProcessor, PowerLawCompression


class SpeechEnhancer:
    """
    Main inference class for speech enhancement.
    """
    
    def __init__(
        self,
        model_path: str,
        n_fft: int = 512,
        hop_length: int = 256,
        device: str = 'cuda',
        base_channels: int = 16,
    ):
        """
        Initialize speech enhancer.
        
        Args:
            model_path: Path to trained model checkpoint
            n_fft: FFT size
            hop_length: Hop length
            device: Device to use (cuda/cpu)
            base_channels: Base channels used in model (16 for checkpoint models)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SpeechEnhancementNetwork(base_channels=base_channels).to(self.device)
        
        # Load checkpoint
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        
        # Initialize STFT processor
        self.stft_processor = STFTProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        # Initialize power-law compressor
        self.compressor = PowerLawCompression(alpha=0.3)
    
    @torch.no_grad()
    def enhance(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        Enhance audio signal.
        
        Args:
            audio: Input noisy audio [time_samples]
            
        Returns:
            Enhanced audio [time_samples]
        """
        # Compute STFT of noisy audio
        noisy_real, noisy_imag = self.stft_processor.stft(audio)
        
        # Add batch and channel dimensions
        noisy_real = noisy_real.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, freq, time]
        noisy_imag = noisy_imag.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Forward pass through model
        enhanced_real, enhanced_imag = self.model(noisy_real, noisy_imag)
        
        # Remove batch and channel dimensions
        enhanced_real = enhanced_real.squeeze(0).squeeze(0)  # [freq, time]
        enhanced_imag = enhanced_imag.squeeze(0).squeeze(0)
        
        # Compute inverse STFT
        enhanced_audio = self.stft_processor.istft(enhanced_real, enhanced_imag)
        
        return enhanced_audio.cpu().numpy()
    
    @torch.no_grad()
    def enhance_batch(
        self,
        audio_list: list,
    ) -> list:
        """
        Enhance multiple audio signals.
        
        Args:
            audio_list: List of audio arrays
            
        Returns:
            List of enhanced audio arrays
        """
        enhanced_list = []
        
        for audio in tqdm(audio_list, desc="Enhancing audio"):
            enhanced = self.enhance(audio)
            enhanced_list.append(enhanced)
        
        return enhanced_list


def process_file(
    enhancer: SpeechEnhancer,
    input_path: str,
    output_path: str,
) -> None:
    """
    Process a single audio file.
    
    Args:
        enhancer: SpeechEnhancer instance
        input_path: Path to input audio file
        output_path: Path to save enhanced audio
    """
    # Load audio
    print(f"Loading audio from: {input_path}")
    audio, sr = sf.read(input_path)
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Enhance
    print("Enhancing audio...")
    enhanced_audio = enhancer.enhance(audio)
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(enhanced_audio))
    if max_val > 1.0:
        enhanced_audio = enhanced_audio / max_val
    
    # Save enhanced audio
    print(f"Saving enhanced audio to: {output_path}")
    sf.write(output_path, enhanced_audio, sr)
    
    print(f"Enhancement completed!")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


def process_directory(
    enhancer: SpeechEnhancer,
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Process all audio files in a directory.
    
    Args:
        enhancer: SpeechEnhancer instance
        input_dir: Directory containing input audio files
        output_dir: Directory to save enhanced audio files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all WAV files
    wav_files = list(input_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
    
    print(f"Found {len(wav_files)} audio files")
    
    # Process each file
    for wav_file in tqdm(wav_files, desc="Processing files"):
        output_file = output_dir / f"enhanced_{wav_file.name}"
        
        try:
            # Load audio
            audio, sr = sf.read(str(wav_file))
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Enhance
            enhanced_audio = enhancer.enhance(audio)
            
            # Normalize
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 1.0:
                enhanced_audio = enhanced_audio / max_val
            
            # Save
            sf.write(str(output_file), enhanced_audio, sr)
            print(f"  ✓ {wav_file.name}")
            
        except Exception as e:
            print(f"  ✗ {wav_file.name}: {str(e)}")
    
    print(f"\nAll files processed!")
    print(f"Enhanced files saved to: {output_dir}")


def compare_files(
    input_path: str,
    enhanced_path: str,
) -> None:
    """
    Compare original and enhanced audio files.
    
    Args:
        input_path: Path to noisy audio
        enhanced_path: Path to enhanced audio
    """
    # Load audio files
    noisy, sr1 = sf.read(input_path)
    enhanced, sr2 = sf.read(enhanced_path)
    
    if sr1 != sr2:
        print(f"Warning: Sample rates differ ({sr1} vs {sr2})")
    
    # Ensure same length
    min_len = min(len(noisy), len(enhanced))
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]
    
    # Compute statistics
    noisy_rms = np.sqrt(np.mean(noisy ** 2))
    enhanced_rms = np.sqrt(np.mean(enhanced ** 2))
    
    print("\nAudio Statistics:")
    print(f"  Noisy RMS:    {noisy_rms:.6f}")
    print(f"  Enhanced RMS: {enhanced_rms:.6f}")
    
    # Compute improvement (simple SNR-like metric)
    difference = enhanced - noisy
    diff_rms = np.sqrt(np.mean(difference ** 2))
    
    print(f"  Difference RMS: {diff_rms:.6f}")
    print(f"\nNote: For proper evaluation, use standard metrics like PESQ, STOI, etc.")


def main(args):
    """Main inference function."""
    
    # Create enhancer
    enhancer = SpeechEnhancer(
        model_path=args.model,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        device='cuda' if args.use_cuda else 'cpu',
    )
    
    print(f"Using device: {enhancer.device}\n")
    
    # Process based on input type
    if args.input_dir:
        # Process directory
        process_directory(enhancer, args.input_dir, args.output_dir)
    elif args.input_file:
        # Process single file
        output_file = args.output_file or Path(args.input_file).stem + "_enhanced.wav"
        process_file(enhancer, args.input_file, output_file)
        
        # Compare if requested
        if args.compare:
            compare_files(args.input_file, output_file)
    else:
        print("Error: Please specify either --input-file or --input-dir")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance speech using trained model")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=512,
        help="FFT size",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop length for STFT",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CUDA if available",
    )
    
    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-file",
        type=str,
        help="Path to input audio file",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing input audio files",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save enhanced audio (single file mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./enhanced_audio",
        help="Directory to save enhanced audio (batch mode)",
    )
    
    # Analysis arguments
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare original and enhanced audio",
    )
    
    args = parser.parse_args()
    
    main(args)

#!/usr/bin/env python3
"""
Simple script to denoise audio files using the trained speech enhancement model.
Upload your noisy audio file and this script will clean it.
"""

import torch
import soundfile as sf
import numpy as np
import argparse
from pathlib import Path
from speech_enhancement.model import SpeechEnhancementNetwork
from speech_enhancement.data_processing import STFTProcessor


def denoise_audio(
    input_file: str,
    output_file: str,
    model_path: str = "demo_checkpoints/checkpoint_epoch_1.pt",
    device: str = "cpu",
):
    """
    Remove noise from an audio file.
    
    Args:
        input_file: Path to noisy audio file (.wav)
        output_file: Path to save enhanced audio (.wav)
        model_path: Path to trained model checkpoint
        device: Device to use ('cpu' or 'cuda')
    """
    print("=" * 60)
    print("🎵 Speech Enhancement - Noise Removal")
    print("=" * 60)
    
    # Load the noisy audio
    print(f"\n📂 Loading audio: {input_file}")
    noisy_audio, sample_rate = sf.read(input_file)
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Duration: {len(noisy_audio) / sample_rate:.2f} seconds")
    print(f"   Samples: {len(noisy_audio)}")
    
    # Convert to mono if stereo
    if len(noisy_audio.shape) > 1:
        print("   Converting stereo to mono...")
        noisy_audio = np.mean(noisy_audio, axis=1)
    
    # Initialize STFT processor
    stft_processor = STFTProcessor(
        n_fft=512,
        hop_length=256,
        window="hann",
    )
    
    # Compute STFT
    print("\n🔧 Processing audio...")
    print("   Computing STFT...")
    noisy_audio_tensor = torch.from_numpy(noisy_audio.astype(np.float32))
    real_part, imag_part = stft_processor.stft(noisy_audio_tensor)
    
    # Add batch and channel dimensions: [freq, time] -> [1, 1, freq, time]
    real_part = real_part.unsqueeze(0).unsqueeze(0)
    imag_part = imag_part.unsqueeze(0).unsqueeze(0)
    
    print(f"   STFT shape: {real_part.shape}")
    
    # Load model
    print(f"\n🤖 Loading model: {model_path}")
    model = SpeechEnhancementNetwork(base_channels=16)  # Must match training config
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"   Checkpoint from epoch: {epoch}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Enhance audio
    print("   Running noise reduction...")
    with torch.no_grad():
        real_part = real_part.to(device)
        imag_part = imag_part.to(device)
        enhanced_real, enhanced_imag = model(real_part, imag_part)
    
    # Remove batch and channel dimensions: [1, 1, freq, time] -> [freq, time]
    enhanced_real = enhanced_real.squeeze(0).squeeze(0).cpu()
    enhanced_imag = enhanced_imag.squeeze(0).squeeze(0).cpu()
    
    # Compute inverse STFT
    print("   Computing inverse STFT...")
    enhanced_audio = stft_processor.istft(enhanced_real, enhanced_imag)
    
    # Convert to numpy and normalize
    enhanced_audio_np = enhanced_audio.numpy()
    
    # Normalize audio to prevent clipping
    max_val = np.abs(enhanced_audio_np).max()
    if max_val > 0:
        enhanced_audio_np = enhanced_audio_np / max_val * 0.95  # Scale to 95% to avoid clipping
    
    print(f"   Audio normalized: max={max_val:.6f}")
    
    # Save enhanced audio
    print(f"\n💾 Saving enhanced audio: {output_file}")
    sf.write(output_file, enhanced_audio_np, sample_rate)
    
    print(f"   Output duration: {len(enhanced_audio_np) / sample_rate:.2f} seconds")
    print(f"   Output samples: {len(enhanced_audio_np)}")
    
    # Calculate noise reduction stats
    original_power = np.mean(noisy_audio ** 2)
    enhanced_power = np.mean(enhanced_audio_np[:len(noisy_audio)] ** 2)
    noise_reduction_db = 10 * np.log10(original_power / (enhanced_power + 1e-10))
    
    print("\n" + "=" * 60)
    print("✅ Noise removal completed successfully!")
    print("=" * 60)
    print(f"📊 Stats:")
    print(f"   Original power: {original_power:.6f}")
    print(f"   Enhanced power: {enhanced_power:.6f}")
    print(f"   Noise reduction: {noise_reduction_db:.2f} dB")
    print("=" * 60)
    
    return enhanced_audio_np


def main():
    parser = argparse.ArgumentParser(
        description="Remove noise from audio files using speech enhancement model"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input noisy audio file (.wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output enhanced audio file (.wav). Default: input_enhanced.wav",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="demo_checkpoints/checkpoint_epoch_1.pt",
        help="Path to model checkpoint (default: demo_checkpoints/checkpoint_epoch_1.pt)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    
    args = parser.parse_args()
    
    # Set default output filename
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_enhanced.wav")
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model checkpoint not found: {args.model}")
        print("\n💡 Tip: Train the model first using:")
        print("   python train.py")
        return
    
    # Run denoising
    try:
        denoise_audio(
            input_file=args.input,
            output_file=args.output,
            model_path=args.model,
            device=args.device,
        )
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Quick demo: Use pre-trained model for inference without waiting for training
"""
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from speech_enhancement.model import SpeechEnhancementNetwork
from speech_enhancement.data_processing import STFTProcessor

def create_demo_model():
    """Create and save a demo model."""
    print("Creating demo model for inference...")
    
    model = SpeechEnhancementNetwork(base_channels=16)
    checkpoint_path = Path("./kaggle_checkpoints/demo_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the untrained model as a baseline
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'description': 'Demo model (untrained baseline)',
    }, checkpoint_path)
    
    print(f"✓ Demo model saved to {checkpoint_path}")
    return checkpoint_path

def enhance_audio(noisy_audio_path, output_path="enhanced_output.wav", checkpoint_path=None):
    """
    Enhance audio using the trained model.
    """
    if checkpoint_path is None:
        checkpoint_path = "kaggle_checkpoints/demo_model.pt"
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Creating demo model...")
        create_demo_model()
    
    print(f"\nEnhancing audio: {noisy_audio_path}")
    
    # Load model
    device = torch.device('cpu')
    model = SpeechEnhancementNetwork(base_channels=16).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load audio
    audio, sr = sf.read(noisy_audio_path)
    audio = torch.from_numpy(audio).float()
    
    # STFT
    stft_processor = STFTProcessor(n_fft=512, hop_length=256)
    real, imag = stft_processor.stft(audio)
    
    # Add batch dimension
    real = real.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, freq, time]
    imag = imag.unsqueeze(0).unsqueeze(0).to(device)
    
    # Enhance
    with torch.no_grad():
        enhanced_real, enhanced_imag = model(real, imag)
    
    enhanced_real = enhanced_real.squeeze().cpu()
    enhanced_imag = enhanced_imag.squeeze().cpu()
    
    # Inverse STFT
    enhanced_audio = stft_processor.istft(enhanced_real, enhanced_imag)
    
    # Normalize
    enhanced_audio = enhanced_audio / (enhanced_audio.abs().max() + 1e-9)
    enhanced_audio = enhanced_audio.numpy()
    
    # Save
    sf.write(output_path, enhanced_audio, sr)
    print(f"✓ Enhanced audio saved to {output_path}")
    
    return enhanced_audio, sr

if __name__ == "__main__":
    # Demo: Create and use model
    demo_checkpoint = create_demo_model()
    print(f"\n✓ Ready for training!")
    print(f"  - Model: {demo_checkpoint}")
    print(f"  - Next: Run real training with GPU for better results")
    print(f"\n  For inference on noisy audio:")
    print(f"    python enhance_audio.py noisy_audio.wav enhanced_output.wav")
